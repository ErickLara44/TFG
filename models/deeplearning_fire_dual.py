# wildfire_dual_pipeline.py
# -*- coding: utf-8 -*-
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# ------------------------------------------------------------
# Utils: device picker (MPS/CUDA/CPU) y métricas básicas
# ------------------------------------------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")
    try:
        ap = average_precision_score(y_true, y_prob)
    except Exception:
        ap = float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "auroc": auroc, "prauc": ap}

# ------------------------------------------------------------
# BLOQUE A — MODELO DE IGNICIÓN (clasificador escalar)
# ResNet + ConvLSTM -> MLP
# ------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4*hidden_dim, kernel_size, padding=pad)

    def forward(self, x, h, c):
        g = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g_ = torch.chunk(g, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g_ = torch.tanh(g_)
        c = f * c + i * g_
        h = o * torch.tanh(c)
        return h, c

class ResNetConvLSTM(nn.Module):
    """Pequeño envoltorio si quieres pasar ya features; aquí usamos directamente C canales."""
    def __init__(self, input_channels: int, hidden_channels: List[int], kernel_size=(3,3), num_layers=1):
        super().__init__()
        assert num_layers == len(hidden_channels)
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            in_ch = input_channels if l == 0 else hidden_channels[l-1]
            self.layers.append(ConvLSTMCell(in_ch, hidden_channels[l], kernel_size[0]))

    def forward(self, x):  # x: [B,T,C,H,W]
        B, T, C, H, W = x.size()
        hs, cs = [], []
        h = [torch.zeros(B, hc, H, W, device=x.device) for hc in [self.layers[0].conv.out_channels//4]]
        c = [torch.zeros_like(h[0])]
        # single-layer for simplicity; extend if needed
        for t in range(T):
            h0, c0 = self.layers[0](x[:, t], h[0], c[0])
            h[0], c[0] = h0, c0
        return [h], [c]  # mimic your original API shape

class SpanishFirePredictionModel(nn.Module):
    """
    IGNICIÓN: devuelve un escalar por clip (probabilidad con sigmoid en inferencia).
    Estructura: ConvLSTM (espacio-temporal) + pooling + MLP.
    (He eliminado la ResNet18 por simplicidad y evitar incompat. de canales;
     si quieres mantenerla, mapea a 3 canales con 1x1 y promedia por tiempo.)
    """
    def __init__(self, num_input_channels: int, temporal_context: int, hidden_dim: int = 128):
        super().__init__()
        self.temporal_context = temporal_context
        self.convlstm = ResNetConvLSTM(
            input_channels=num_input_channels, hidden_channels=[hidden_dim], kernel_size=(3,3), num_layers=1
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)  # logit
        )

    def forward(self, x):  # x: [B,T,C,H,W]
        h_last = self.convlstm(x)[0][-1]        # [B, hidden, H, W]
        pooled = self.pool(h_last).flatten(1)   # [B, hidden]
        logit  = self.classifier(pooled).squeeze(1)  # [B]
        return logit

# ------------------------------------------------------------
# BLOQUE B — MODELO DE PROPAGACIÓN (mapa prob. t+1)
# ConvLSTM -> Conv 1x1 (sigmoid) ; Pérdida BCE + FSS
# ------------------------------------------------------------
class FireSpreadConvLSTM(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_dim, kernel_size=3)
        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):  # x: [B,T,C,H,W]
        B,T,C,H,W = x.size()
        h = torch.zeros(B, 64, H, W, device=x.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        prob = torch.sigmoid(self.head(h))  # [B,1,H,W]
        return prob

class FSSLoss(nn.Module):
    """FSS diferenciable: media en ventanas con AvgPool y MSE normalizado."""
    def __init__(self, scales: List[int] = [3,5,9]):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B,1,H,W] (0..1), target: [B,1,H,W] (0/1)
        total = 0.0
        for s in self.scales:
            pool = nn.AvgPool2d(s, stride=1, padding=s//2)
            pf = pool(pred)
            tf = pool(target.float())
            mse = F.mse_loss(pf, tf, reduction="mean")
            ref = F.mse_loss(tf, torch.zeros_like(tf), reduction="mean") + 1e-6
            fss = 1.0 - mse/ref
            total += (1.0 - fss)
        return total / len(self.scales)

# ------------------------------------------------------------
# DATASETS (esqueletos) — Conecta aquí tu datacubo IberFire
# ------------------------------------------------------------
class IgnitionDataset(Dataset):
    """
    Debe devolver:
      x_seq: [T, C, H, W]  (p.ej., T=7 días, C=canales: meteo+topo+fuel+estado)
      y:     escalar (0/1)  -> ¿habrá incendio ≥30ha?
    """
    def __init__(self, index_list: List, transform=None):
        self.index_list = index_list
        self.transform = transform

    def __len__(self): return len(self.index_list)

    def __getitem__(self, i):
        # TODO: extrae de tu datacubo
        # x_seq_np = ...  # shape (T,C,H,W)
        # y = ...         # 0/1
        raise NotImplementedError

class SpreadDataset(Dataset):
    """
    Debe devolver:
      x_seq: [T, C, H, W] (incluye estado_t, meteo, topografía, fuel...)
      y_map: [1, H, W]  (binario: quemado en t+1)
    """
    def __init__(self, index_list: List, transform=None):
        self.index_list = index_list
        self.transform = transform

    def __len__(self): return len(self.index_list)

    def __getitem__(self, i):
        # TODO: extrae de tu datacubo
        # x_seq_np = ...  # (T,C,H,W)
        # y_map_np = ...  # (1,H,W)
        raise NotImplementedError

# ------------------------------------------------------------
# ENTRENADORES SENCILLOS (sin Lightning), con validación
# ------------------------------------------------------------
def train_ignition(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    device: Optional[torch.device] = None
):
    device = device or pick_device()
    model.to(device)

    # pos_weight para clases desbalanceadas (en el dataset de entrenamiento)
    fire_count = 0
    total = 0
    for _, y in train_loader:
        total += y.numel()
        fire_count += int(y.sum().item())
    pos_weight = torch.tensor([(total - fire_count) / max(fire_count, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=True)

    best_f1 = -1.0
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device)  # [B,T,C,H,W]
            y = y.to(device).float()  # [B]
            opt.zero_grad()
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

        # validación
        model.eval()
        val_loss = 0.0
        y_true, y_prob = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device).float()
                logit = model(x)
                loss = criterion(logit, y)
                val_loss += loss.item()
                prob = torch.sigmoid(logit).detach().cpu().numpy()
                y_prob.extend(prob.tolist())
                y_true.extend(y.cpu().numpy().tolist())
        y_true = np.array(y_true).astype(int)
        y_prob = np.array(y_prob)
        metrics = binary_classification_metrics(y_true, y_prob, thr=0.5)
        sched.step(val_loss / max(1, len(val_loader)))

        print(f"[IGN] Epoch {ep:03d} | TrainLoss {running/len(train_loader):.4f} | "
              f"ValLoss {val_loss/len(val_loader):.4f} | F1 {metrics['f1']:.3f} | "
              f"AUROC {metrics['auroc']:.3f} | PRAUC {metrics['prauc']:.3f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"state_dict": model.state_dict(), "best_f1": best_f1}, "best_ignition.pth")
            print("✅ [IGN] Mejor modelo guardado.")

def train_spread(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    device: Optional[torch.device] = None,
    fss_scales: List[int] = [3,5,9],
):
    device = device or pick_device()
    model.to(device)

    bce = nn.BCEWithLogitsLoss()  # usaremos sobre logits si modificas salida; aquí ya usamos prob, así que BCE normal
    fss_loss_fn = FSSLoss(scales=fss_scales)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=True)

    def combined_loss(prob, y):
        # prob ya en [0,1]; si quieres logits, cambia por BCEWithLogitsLoss
        bce_loss = F.binary_cross_entropy(prob, y.float())
        fss = fss_loss_fn(prob, y)
        return bce_loss + fss, bce_loss.item(), fss.item()

    best_val = 1e9
    for ep in range(1, epochs+1):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)           # [B,T,C,H,W]
            y = y.to(device)           # [B,1,H,W] 0/1
            opt.zero_grad()
            prob = model(x)            # [B,1,H,W], sigmoid
            loss, bce_l, fss_l = combined_loss(prob, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()

        # Validación (reportar FSS promedio)
        model.eval()
        val_loss, val_bce, val_fss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                prob = model(x)
                loss, bce_l, fss_l = combined_loss(prob, y)
                val_loss += loss.item()
                val_bce += bce_l
                val_fss += fss_l

        val_loss /= max(1, len(val_loader))
        val_bce  /= max(1, len(val_loader))
        val_fss  /= max(1, len(val_loader))
        sched.step(val_loss)

        print(f"[SPR] Epoch {ep:03d} | TrainLoss {run_loss/len(train_loader):.4f} | "
              f"ValLoss {val_loss:.4f} | BCE {val_bce:.4f} | FSSloss {val_fss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "val_loss": best_val}, "best_spread.pth")
            print("✅ [SPR] Mejor modelo guardado.")

# ------------------------------------------------------------
# PIPELINE ORQUESTADOR: ignición -> propagación (condicionada)
# ------------------------------------------------------------
@torch.no_grad()
def dual_inference(
    ignition_model: nn.Module,
    spread_model: nn.Module,
    clip_seq: torch.Tensor,         # [1,T,C,H,W]   — tu ventana
    env_extra: Optional[torch.Tensor] = None,  # [1,T,C2,H,W]  opcional
    ignition_thr: float = 0.5,
    device: Optional[torch.device] = None
) -> Tuple[float, Optional[torch.Tensor]]:
    device = device or pick_device()
    ignition_model.eval().to(device)
    spread_model.eval().to(device)

    x = clip_seq.to(device)
    p_fire = torch.sigmoid(ignition_model(x)).item()  # escalar
    spread_map = None
    if p_fire >= ignition_thr:
        if env_extra is not None:
            x_spread = torch.cat([x, env_extra.to(device)], dim=2)  # concat canales
        else:
            x_spread = x
        spread_map = spread_model(x_spread)  # [1,1,H,W] prob
    return p_fire, spread_map

# ------------------------------------------------------------
# EJEMPLO DE CONFIGURACIÓN
# ------------------------------------------------------------
def example_setup():
    """
    Ejemplo de cómo instanciar modelos y dataloaders.
    Sustituye IgnitionDataset/SpreadDataset por tus clases con IberFire.
    """
    T = 7
    C_ign = 12   # meteorología + estado + etc (ajusta)
    C_spr = 16   # ignición + meteo + DEM + fuel (ajusta)
    H = W = 128

    ign_model = SpanishFirePredictionModel(num_input_channels=C_ign, temporal_context=T, hidden_dim=128)
    spr_model = FireSpreadConvLSTM(input_channels=C_spr, hidden_dim=64)

    # Datasets/dataloaders (placeholders)
    train_ign = IgnitionDataset(index_list=[])
    val_ign   = IgnitionDataset(index_list=[])
    train_spr = SpreadDataset(index_list=[])
    val_spr   = SpreadDataset(index_list=[])

    dl_tr_ign = DataLoader(train_ign, batch_size=8, shuffle=True, num_workers=2)
    dl_va_ign = DataLoader(val_ign,   batch_size=8, shuffle=False, num_workers=2)
    dl_tr_spr = DataLoader(train_spr, batch_size=4, shuffle=True, num_workers=2)
    dl_va_spr = DataLoader(val_spr,   batch_size=4, shuffle=False, num_workers=2)

    # Entrenamientos
    # train_ignition(ign_model, dl_tr_ign, dl_va_ign, epochs=30, lr=1e-3)
    # train_spread(spr_model, dl_tr_spr, dl_va_spr, epochs=30, lr=1e-3)

    # Inference orquestado (ejemplo con tensores aleatorios)
    clip = torch.rand(1, T, C_ign, H, W)
    env  = torch.rand(1, T, max(0, C_spr - C_ign), H, W) if C_spr > C_ign else None
    p_fire, spread = dual_inference(ign_model, spr_model, clip, env, ignition_thr=0.5)
    print("Prob. ignición:", p_fire, "| spread_map:", None if spread is None else spread.shape)

if __name__ == "__main__":
    # example_setup()
    pass