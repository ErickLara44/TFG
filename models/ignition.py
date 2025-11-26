import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

# ========================================
# COMPONENTES ARQUITECTÓNICOS ROBUSTOS
# ========================================

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell mejorada con normalización y regularización
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, 
            kernel_size, padding=pad, bias=False
        )
        
        # Normalización y regularización
        self.layer_norm = nn.GroupNorm(4, 4 * hidden_dim)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        gates = self.layer_norm(gates)
        gates = self.dropout(gates)
        
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c = f * c + i * g
        h = o * torch.tanh(c)
        
        return h, c


class MultiScaleConvLSTM(nn.Module):
    """
    ConvLSTM multi-escala para capturar patrones a diferentes resoluciones
    """
    def __init__(self, input_channels, hidden_dims=[64, 128], dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Primera capa
        self.layers.append(ConvLSTMCell(input_channels, hidden_dims[0], dropout=dropout))
        
        # Capas adicionales con downsampling
        for i in range(1, len(hidden_dims)):
            self.layers.append(ConvLSTMCell(hidden_dims[i-1], hidden_dims[i], dropout=dropout))
        
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.hidden_dims = hidden_dims
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        
        # Estados iniciales para cada escala
        h_states = []
        c_states = []
        for hidden_dim in self.hidden_dims:
            current_h, current_w = H, W
            if len(h_states) > 0:  # Downsampling para escalas superiores
                current_h, current_w = current_h // (2 ** len(h_states)), current_w // (2 ** len(h_states))
            
            h_states.append(torch.zeros(B, hidden_dim, current_h, current_w, device=x.device))
            c_states.append(torch.zeros(B, hidden_dim, current_h, current_w, device=x.device))
        
        # Procesar secuencia temporal
        outputs = []
        for t in range(T):
            current_input = x[:, t]  # (B, C, H, W)
            
            # Procesar cada escala
            for layer_idx, layer in enumerate(self.layers):
                if layer_idx > 0:
                    current_input = self.downsample(current_input)
                
                h_states[layer_idx], c_states[layer_idx] = layer(
                    current_input, h_states[layer_idx], c_states[layer_idx]
                )
                current_input = h_states[layer_idx]
            
            outputs.append(h_states[-1])  # Usar la última escala
        
        return torch.stack(outputs, dim=1)  # (B, T, hidden_dim, H_final, W_final)


class SpatialAttention(nn.Module):
    """
    Módulo de atención espacial para enfocarse en regiones críticas
    """
    def __init__(self, channels):
        super().__init__()
        self.conv_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_map = self.conv_attention(x)
        return x * attention_map, attention_map


class TemporalAttention(nn.Module):
    """
    Módulo de atención temporal para ponderar la importancia de diferentes timesteps
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x: (B, T, features)
        B, T, features = x.size()
        attention_weights = self.attention(x.view(-1, features)).view(B, T, 1)
        attended = (x * attention_weights).sum(dim=1)
        return attended, attention_weights


# ========================================
# MODELO PRINCIPAL ROBUSTO
# ========================================

class RobustFireIgnitionModel(nn.Module):
    """
    Modelo robusto para predicción de ignición con múltiples componentes avanzados:
    - ConvLSTM multi-escala
    - Atención espacial y temporal  
    - Regularización avanzada
    - Múltiples cabezas de predicción
    """
    def __init__(self, num_input_channels, temporal_context, hidden_dims=[64, 128], dropout=0.2):
        super().__init__()
        self.temporal_context = temporal_context
        self.num_input_channels = num_input_channels
        
        # 1. ConvLSTM multi-escala
        self.multi_scale_lstm = MultiScaleConvLSTM(
            num_input_channels, hidden_dims, dropout=dropout
        )
        
        # 2. Atención espacial
        self.spatial_attention = SpatialAttention(hidden_dims[-1])
        
        # 3. Pooling adaptativo multi-escala
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),    # Global
            nn.AdaptiveAvgPool2d((2, 2)),    # Regional  
            nn.AdaptiveAvgPool2d((4, 4))     # Local
        ])
        
        # 4. Dimensiones de características combinadas
        total_pool_features = hidden_dims[-1] * (1 + 4 + 16)  # 1x1 + 2x2 + 4x4
        
        # 5. Atención temporal
        self.temporal_attention = TemporalAttention(total_pool_features)
        
        # 6. Extractor de características robustas
        self.feature_extractor = nn.Sequential(
            nn.Linear(total_pool_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
        )
        
        # 7. Cabezas de predicción múltiple
        self.ignition_head = nn.Linear(256, 1)      # Probabilidad principal
        self.confidence_head = nn.Linear(256, 1)    # Confianza de la predicción
        self.risk_head = nn.Linear(256, 3)          # Riesgo: bajo, medio, alto
        
        # 8. Inicialización de pesos
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Inicialización mejorada de pesos"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        
        # 1. ConvLSTM multi-escala
        lstm_output = self.multi_scale_lstm(x)  # (B, T, hidden_dim, H', W')
        
        # 2. Procesar cada timestep con atención espacial
        attended_features = []
        attention_maps = []
        
        for t in range(T):
            timestep_features = lstm_output[:, t]  # (B, hidden_dim, H', W')
            
            # Atención espacial
            attended_feat, att_map = self.spatial_attention(timestep_features)
            attention_maps.append(att_map)
            
            # Pooling multi-escala
            pooled_features = []
            for pool in self.adaptive_pools:
                pooled = pool(attended_feat)
                pooled_features.append(pooled.flatten(1))
            
            # Concatenar características de diferentes escalas
            combined_features = torch.cat(pooled_features, dim=1)
            attended_features.append(combined_features)
        
        # 3. Atención temporal
        temporal_features = torch.stack(attended_features, dim=1)  # (B, T, features)
        final_features, temporal_weights = self.temporal_attention(temporal_features)
        
        # 4. Extracción final de características
        robust_features = self.feature_extractor(final_features)
        
        # 5. Predicciones múltiples
        ignition_logit = self.ignition_head(robust_features)
        confidence_logit = self.confidence_head(robust_features)
        risk_logits = self.risk_head(robust_features)
        
        outputs = {
            'ignition': ignition_logit.squeeze(1),
            'confidence': torch.sigmoid(confidence_logit.squeeze(1)),
            'risk_distribution': F.softmax(risk_logits, dim=1)
        }
        
        if return_attention:
            outputs['spatial_attention'] = torch.stack(attention_maps, dim=1)
            outputs['temporal_attention'] = temporal_weights
        
        return outputs


# ========================================
# FUNCIÓN DE ENTRENAMIENTO AVANZADA
# ========================================

def train_robust_ignition_model(
    model, train_loader, val_loader,
    epochs=50, lr=0.001, device='auto',
    save_metrics_path="robust_ignition_metrics.json",
    use_mixed_precision=True,
    early_stopping_patience=10
):
    """
    Entrenamiento robusto con múltiples mejoras:
    - Mixed precision training
    - Early stopping  
    - Múltiples pérdidas
    - Schedulers adaptativos
    """
    
    # Selección automática de dispositivo
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    print(f"Usando dispositivo: {device}")
    model = model.to(device)
    
    # Configuración de entrenamiento
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8
    )
    
    # Schedulers combinados
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, verbose=True
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    
    # Pesos para clases desbalanceadas
    fire_count = sum(1 for _, label in train_loader.dataset if label.item() > 0.5)
    total_count = len(train_loader.dataset)
    pos_weight = torch.tensor([(total_count - fire_count) / max(fire_count, 1)]).to(device)
    
    # Funciones de pérdida
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # Tracking
    best_f1 = 0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [], "f1": [], "auroc": [], 
        "precision": [], "recall": [], "lr": []
    }
    
    for epoch in range(epochs):
        # ENTRENAMIENTO
        model.train()
        train_losses = []
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    
                    # Pérdida principal (ignición)
                    loss_ignition = bce_loss(outputs['ignition'], target.squeeze())
                    
                    # Pérdida de confianza (target = 1 para todas las predicciones)
                    loss_confidence = mse_loss(outputs['confidence'], torch.ones_like(outputs['confidence']))
                    
                    # Pérdida combinada
                    total_loss = loss_ignition + 0.1 * loss_confidence
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss_ignition = bce_loss(outputs['ignition'], target.squeeze())
                loss_confidence = mse_loss(outputs['confidence'], torch.ones_like(outputs['confidence']))
                total_loss = loss_ignition + 0.1 * loss_confidence
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_losses.append(total_loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # VALIDACIÓN
        model.eval()
        val_losses = []
        all_targets, all_probs = [], []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device, dtype=torch.float32)
                
                outputs = model(data)
                loss = bce_loss(outputs['ignition'], target.squeeze())
                val_losses.append(loss.item())
                
                probs = torch.sigmoid(outputs['ignition'])
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        
        # MÉTRICAS
        all_targets = np.array(all_targets).flatten()
        all_probs = np.array(all_probs)
        
        # F1 optimizado
        precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        best_f1_current = f1_scores[best_threshold_idx]
        best_threshold = thresholds[best_threshold_idx]
        
        # Otras métricas
        pred_binary = (all_probs > best_threshold).astype(int)
        precision_final = precision[best_threshold_idx]
        recall_final = recall[best_threshold_idx]
        
        try:
            auroc = roc_auc_score(all_targets, all_probs)
        except:
            auroc = 0.0
        
        # Actualizar schedulers
        scheduler_plateau.step(avg_val_loss)
        scheduler_cosine.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Guardar métricas
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["f1"].append(best_f1_current)
        history["auroc"].append(auroc)
        history["precision"].append(precision_final)
        history["recall"].append(recall_final)
        history["lr"].append(current_lr)
        
        print(f"[ROBUST] Epoch {epoch+1:03d}/{epochs} | "
              f"TrLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | "
              f"F1: {best_f1_current:.3f} | AUROC: {auroc:.3f} | "
              f"Thr: {best_threshold:.3f} | LR: {current_lr:.2e}")
        
        # Early stopping y mejor modelo
        if best_f1_current > best_f1:
            best_f1 = best_f1_current
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': best_f1,
                'threshold': best_threshold,
                'epoch': epoch
            }, 'best_robust_ignition_model.pth')
            print(f"✅ Mejor modelo guardado (F1: {best_f1:.3f})")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Early stopping en epoch {epoch+1}")
            break
    
    # Guardar historial
    with open(save_metrics_path, "w") as f:
        json.dump(history, f, indent=4)
    
    print(f"📊 Entrenamiento completado. Mejor F1: {best_f1:.3f}")
    return history

print("✅ Modelo robusto de ignición implementado!")
print("🚀 Incluye: ConvLSTM multi-escala, atención espacial/temporal, regularización avanzada")