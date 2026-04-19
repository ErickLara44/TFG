# Modelo de Propagación v1 — Resultados

**Fecha:** 2026-02-26  
**Modelo guardado:** `models/spread_v1_valIoU0536_testIoU0289.pth`

---

## Configuración

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | `RobustFireSpreadModel` (ConvLSTM multi-escala) |
| `hidden_dims` | `[64, 128]` |
| Input channels | 12 (11 features + fire_state) |
| Crop size | 32×32 px (centrado en fuego, on-the-fly desde 224×224) |
| Contexto temporal | 3 días |
| Epochs | 30 |
| Batch size | 8 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Loss | FocalDice (alpha=0.9, gamma=2.0) |
| Augmentación | 8 transforms on-the-fly + corrección vector viento |
| Device | MPS (Apple Silicon) |

## Dataset

| Split | Muestras | Años |
|-------|----------|------|
| Train | 468 | 2007–2020 |
| Val | 199 | 2021–2022 |
| Test | 159 | 2023–2024 |

> Solo muestras donde el fuego se propagó (`y_sum > 0`). Sin data leakage temporal.

## Resultados por Epoch (selección)

| Epoch | Train Loss | Train IoU | Val Loss | Val IoU |
|-------|-----------|-----------|----------|---------|
| 1 | ~1.10 | ~0.010 | — | ~0.010 |
| 10 | 0.944 | 0.056 | 0.926 | 0.075 |
| 15 | 0.660 | 0.202 | 0.580 | 0.302 |
| 20 | 0.518 | 0.390 | 0.450 | 0.486 |
| 24 | 0.521 | 0.390 | 0.436 | 0.513 |
| **28** ⭐ | **0.446** | **0.446** | **0.397** | **0.536** |
| 29 | 0.442 | 0.449 | 0.430 | 0.502 |
| 30 | 0.446 | 0.454 | 0.461 | 0.465 |

## Evaluación en Test (mejor modelo — epoch 28)

| Métrica | Valor |
|---------|-------|
| **IoU** | **0.2884** |
| **F1** | **0.6526** |
| Precision | 0.7047 |
| Recall | 0.6077 |
| Píxeles fuego reales | 1119 |
| Píxeles fuego predichos | 965 |

## Análisis

- **Gap val→test** (0.536 → 0.289): el modelo generaliza razonablemente pero sufre por distribución temporal (train 2007-2020, test 2023-2024) y por entrenarse solo con positivos.
- **Precision > Recall**: el modelo es conservador — prefiere no predecir fuego a tener falsos positivos.
- **Dataset pequeño**: 468 muestras train limita la generalización.

## Próximos pasos (v2)

- [ ] **Negative mining**: incluir muestras `y_sum == 0` (~50% ratio) → enseñar cuándo NO propagar
- [ ] **WeightedRandomSampler**: balancear pos/neg en cada batch
- [ ] **LR scheduler cosine**: refinado más suave en epochs altas
