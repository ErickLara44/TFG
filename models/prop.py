from typing import List, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json

# ========================================
# COMPONENTES ARQUITECTÓNICOS AVANZADOS
# ========================================

class EnhancedConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell mejorada con normalización, regularización y conexiones residuales
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        pad = kernel_size // 2
        # Gates principales
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, 
            kernel_size, padding=pad, bias=False
        )
        
        # Normalización y regularización
        self.layer_norm = nn.GroupNorm(4, 4 * hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout2d(dropout)
        
        # Conexión residual si las dimensiones coinciden
        self.use_residual = (input_dim == hidden_dim)
        if not self.use_residual:
            self.residual_proj = nn.Conv2d(input_dim, hidden_dim, 1, bias=False)
        
    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv_gates(combined)
        gates = self.layer_norm(gates)
        gates = self.dropout(gates)
        
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)  
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_new = f * c + i * g
        
        # Conexión residual en el estado cell
        if self.use_residual:
            c_new = c_new + x
        else:
            c_new = c_new + self.residual_proj(x)
            
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class MultiScaleFireConvLSTM(nn.Module):
    """
    ConvLSTM multi-escala específico para propagación de fuego
    Captura patrones de propagación a diferentes resoluciones espaciales
    """
    def __init__(self, input_channels, hidden_dims=[32, 64, 128], dropout=0.1):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.num_scales = len(hidden_dims)
        
        # Capas ConvLSTM para cada escala
        self.lstm_cells = nn.ModuleList()
        current_input_dim = input_channels
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.lstm_cells.append(
                EnhancedConvLSTMCell(current_input_dim, hidden_dim, dropout=dropout)
            )
            current_input_dim = hidden_dim
        
        # Pooling y upsampling para escalas múltiples
        self.downsample = nn.MaxPool2d(2, stride=2)  # MaxPool mejor para fuego (mantiene edges)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        
        # Estados iniciales multi-escala
        h_states = []
        c_states = []
        
        current_h, current_w = H, W
        for i, hidden_dim in enumerate(self.hidden_dims):
            if i > 0:
                current_h, current_w = current_h // 2, current_w // 2
                
            h_states.append(torch.zeros(B, hidden_dim, current_h, current_w, device=x.device))
            c_states.append(torch.zeros(B, hidden_dim, current_h, current_w, device=x.device))
        
        # Secuencias de salida para cada escala
        outputs_multiscale = []
        
        for t in range(T):
            current_input = x[:, t]  # (B, C, H, W)
            scale_outputs = []
            
            for scale_idx, lstm_cell in enumerate(self.lstm_cells):
                # Downsample para escalas superiores
                if scale_idx > 0:
                    current_input = self.downsample(current_input)
                
                # Procesar con ConvLSTM
                h_states[scale_idx], c_states[scale_idx] = lstm_cell(
                    current_input, h_states[scale_idx], c_states[scale_idx]
                )
                
                scale_outputs.append(h_states[scale_idx])
                current_input = h_states[scale_idx]
            
            outputs_multiscale.append(scale_outputs)
        
        return outputs_multiscale  # Lista de [timesteps][escalas]


class FirePropagationAttention(nn.Module):
    """
    Módulo de atención específico para propagación de fuego
    Se enfoca en bordes de fuego y direcciones de viento
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        
        # Atención espacial para bordes de fuego
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels // reduction, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
        # Atención de canal (qué tipo de información es más importante)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Atención espacial
        spatial_att = self.spatial_attention(x)
        
        # Atención de canal
        channel_att = self.channel_attention(x)
        
        # Combinar ambas atenciones
        attended = x * spatial_att * channel_att
        
        return attended, spatial_att


class DirectionalConvBlock(nn.Module):
    """
    Bloque convolucional que captura direccionalidad en la propagación
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Convoluciones direccionales (horizontal, vertical, diagonales)
        self.conv_horizontal = nn.Conv2d(in_channels, out_channels//4, (1,3), padding=(0,1))
        self.conv_vertical = nn.Conv2d(in_channels, out_channels//4, (3,1), padding=(1,0))
        self.conv_diag1 = nn.Conv2d(in_channels, out_channels//4, 3, padding=1)  # Normal 3x3
        self.conv_diag2 = nn.Conv2d(in_channels, out_channels//4, 3, padding=1, dilation=2)  # Dilated
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Procesar diferentes direcciones
        h_feat = self.conv_horizontal(x)
        v_feat = self.conv_vertical(x)
        d1_feat = self.conv_diag1(x)
        d2_feat = self.conv_diag2(x)
        
        # Concatenar características direccionales
        combined = torch.cat([h_feat, v_feat, d1_feat, d2_feat], dim=1)
        
        return self.activation(self.norm(combined))


# ========================================
# MODELO PRINCIPAL ROBUSTO DE PROPAGACIÓN
# ========================================

class RobustFireSpreadModel(nn.Module):
    """
    Modelo robusto para predicción de propagación con:
    - ConvLSTM multi-escala
    - Atención específica para fuego
    - Análisis direccional
    - Múltiples cabezas de predicción
    """
    def __init__(self, input_channels, hidden_dims=[32, 64, 128], dropout=0.15):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        
        # 1. Encoder multi-escala
        self.multiscale_lstm = MultiScaleFireConvLSTM(
            input_channels, hidden_dims, dropout=dropout
        )
        
        # 2. Procesamiento direccional
        self.directional_block = DirectionalConvBlock(
            hidden_dims[-1], hidden_dims[-1]
        )
        
        # 3. Módulo de atención
        self.fire_attention = FirePropagationAttention(hidden_dims[-1])
        
        # 4. Decoder multi-escala (para reconstruir resolución original)
        self.decoder = nn.ModuleList()
        for i in range(len(hidden_dims)-1, 0, -1):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 3, 
                                 stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i-1]),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ))
        
        # 5. Cabezas de predicción múltiple
        final_channels = hidden_dims[0]
        
        # Cabeza principal: probabilidad de propagación
        self.spread_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels//2, 1, 1),
            nn.Sigmoid()
        )
        
        # Cabeza de intensidad: qué tan intenso será el fuego
        self.intensity_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels//2, 1, 1),
            nn.Sigmoid()
        )
        
        # Cabeza direccional: vector de propagación dominante
        self.direction_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels//2, 2, 1),  # dx, dy
            nn.Tanh()  # Valores entre -1 y 1
        )
        
        # Inicialización de pesos
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        
        # 1. Encoder multi-escala ConvLSTM
        multiscale_outputs = self.multiscale_lstm(x)  # [timesteps][escalas]
        
        # Usar la última timestep de la escala más alta
        final_features = multiscale_outputs[-1][-1]  # Última timestep, última escala
        
        # 2. Procesamiento direccional
        directional_features = self.directional_block(final_features)
        
        # 3. Atención específica para fuego
        attended_features, attention_map = self.fire_attention(directional_features)
        
        # 4. Decoder para reconstruir resolución original
        decoded = attended_features
        for decoder_layer in self.decoder:
            decoded = decoder_layer(decoded)
        
        # 5. Predicciones múltiples
        spread_prob = self.spread_head(decoded)
        intensity = self.intensity_head(decoded)  
        direction_vec = self.direction_head(decoded)
        
        outputs = {
            'spread_probability': spread_prob,
            'fire_intensity': intensity,
            'propagation_direction': direction_vec
        }
        
        if return_attention:
            outputs['attention_map'] = attention_map
            outputs['multiscale_features'] = multiscale_outputs
        
        return outputs


# ========================================
# PÉRDIDAS AVANZADAS
# ========================================

class AdvancedFSSLoss(nn.Module):
    """
    FSS Loss mejorada con ponderación adaptativa
    """
    def __init__(self, scales=[3, 5, 9, 15], weights=None):
        super().__init__()
        self.scales = scales
        self.weights = weights or [1.0] * len(scales)
        
    def forward(self, pred, target):
        total_loss = 0.0
        total_weight = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            pool = nn.AvgPool2d(scale, stride=1, padding=scale//2)
            
            pred_pooled = pool(pred)
            target_pooled = pool(target.float())
            
            mse_num = F.mse_loss(pred_pooled, target_pooled, reduction="mean")
            mse_den = F.mse_loss(target_pooled, torch.zeros_like(target_pooled), reduction="mean")
            
            fss = 1.0 - mse_num / (mse_den + 1e-8)
            scale_loss = (1.0 - fss) * weight
            
            total_loss += scale_loss
            total_weight += weight
        
        return total_loss / total_weight


class DirectionalConsistencyLoss(nn.Module):
    """
    Pérdida para asegurar consistencia direccional en la propagación
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, direction_pred, spread_pred, spread_target):
        # Solo calcular pérdida donde hay propagación real
        fire_mask = (spread_target > 0.5).float()
        
        if fire_mask.sum() < 1:
            return torch.tensor(0.0, device=direction_pred.device)
        
        # Calcular gradiente del fuego real (dirección verdadera)
        grad_y, grad_x = torch.gradient(spread_target.squeeze(1))
        true_direction = torch.stack([grad_x, grad_y], dim=1)  # (B, 2, H, W)
        
        # Normalizar direcciones
        direction_pred_norm = F.normalize(direction_pred, p=2, dim=1)
        true_direction_norm = F.normalize(true_direction, p=2, dim=1)
        
        # Pérdida coseno (1 - similitud coseno)
        cosine_sim = (direction_pred_norm * true_direction_norm).sum(dim=1, keepdim=True)
        directional_loss = (1 - cosine_sim) * fire_mask
        
        return directional_loss.mean()


# ========================================
# ENTRENAMIENTO AVANZADO
# ========================================

def train_robust_spread_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'auto',
    save_metrics_path: str = "robust_spread_metrics.json",
    use_mixed_precision: bool = True
):
    """
    Entrenamiento robusto con múltiples pérdidas y regularización avanzada
    """
    
    # Dispositivo
    if device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() 
                            else 'cuda' if torch.cuda.is_available() 
                            else 'cpu')
    
    print(f"Usando dispositivo: {device}")
    model.to(device)
    
    # Optimizador y scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*10, epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    
    # Pérdidas avanzadas
    bce_loss = nn.BCELoss()
    fss_loss = AdvancedFSSLoss(scales=[3, 5, 9, 15])
    directional_loss = DirectionalConsistencyLoss()
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    
    # Tracking
    best_iou = 0.0
    history = {
        "train_loss": [], "val_loss": [], "iou": [], "dice": [], 
        "pixel_acc": [], "directional_error": []
    }
    
    for epoch in range(epochs):
        # ENTRENAMIENTO
        model.train()
        train_losses = []
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(x)
                    
                    # Pérdidas múltiples
                    loss_bce = bce_loss(outputs['spread_probability'], y)
                    loss_fss = fss_loss(outputs['spread_probability'], y)
                    loss_dir = directional_loss(outputs['propagation_direction'], 
                                              outputs['spread_probability'], y)
                    
                    # Pérdida total ponderada
                    total_loss = loss_bce + 0.3 * loss_fss + 0.1 * loss_dir
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x)
                loss_bce = bce_loss(outputs['spread_probability'], y)
                loss_fss = fss_loss(outputs['spread_probability'], y)
                loss_dir = directional_loss(outputs['propagation_direction'], 
                                          outputs['spread_probability'], y)
                total_loss = loss_bce + 0.3 * loss_fss + 0.1 * loss_dir
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            train_losses.append(total_loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # VALIDACIÓN
        model.eval()
        val_losses = []
        metrics_all = {"iou": [], "dice": [], "pixel_acc": []}
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                outputs = model(x)
                
                # Pérdida de validación (solo principal)
                val_loss = bce_loss(outputs['spread_probability'], y)
                val_losses.append(val_loss.item())
                
                # Métricas de segmentación
                pred = outputs['spread_probability'].cpu()
                target = y.cpu()
                
                metrics = segmentation_metrics(pred, target, thr=0.5)
                for k, v in metrics.items():
                    metrics_all[k].append(v)
        
        avg_val_loss = np.mean(val_losses)
        avg_metrics = {k: np.mean(v) for k, v in metrics_all.items()}
        
        # Guardar historial
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["iou"].append(avg_metrics["iou"])
        history["dice"].append(avg_metrics["dice"])
        history["pixel_acc"].append(avg_metrics["pixel_acc"])
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[ROBUST SPREAD] Epoch {epoch+1:03d}/{epochs} | "
              f"TrLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | "
              f"IoU: {avg_metrics['iou']:.3f} | Dice: {avg_metrics['dice']:.3f} | "
              f"LR: {current_lr:.2e}")
        
        # Guardar mejor modelo
        if avg_metrics["iou"] > best_iou:
            best_iou = avg_metrics["iou"]
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou_score': best_iou,
                'epoch': epoch
            }, 'best_robust_spread_model.pth')
            print(f"✅ Mejor modelo guardado (IoU: {best_iou:.3f})")
    
    # Guardar historial
    with open(save_metrics_path, "w") as f:
        json.dump(history, f, indent=4)
    
    print(f"📊 Entrenamiento completado. Mejor IoU: {best_iou:.3f}")
    return history


# Métrica de segmentación mejorada
@torch.no_grad()
def segmentation_metrics(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    """Métricas de segmentación robustas"""
    pred_bin = (pred >= thr).int()
    target = target.int()
    
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    tn = ((1 - pred_bin) * (1 - target)).sum().item()

    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        "iou": iou, "dice": dice, "pixel_acc": acc,
        "precision": precision, "recall": recall
    }

print("✅ Modelo robusto de propagación implementado!")
print("🔥 Incluye: ConvLSTM multi-escala, atención para fuego, análisis direccional, pérdidas avanzadas")