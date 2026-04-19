
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os

class PatchDataset(Dataset):
    """
    Dataset optimizado que carga parches pre-calculados (.pt).
    Mucho más rápido que leer NetCDF aleatoriamente.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.files = sorted(list(self.data_dir.glob("*.pt")))
        
        if not self.files:
            print(f"⚠️ No se encontraron archivos .pt en {self.data_dir}")
        else:
            print(f"📦 PatchDataset cargado desde {self.data_dir}: {len(self.files)} muestras.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            # Cargar con weights_only=True por seguridad y rapidez si es posible en versiones nuevas
            # Si da error en torch viejo, quitar weights_only
            data = torch.load(file_path, weights_only=False) 
            x = data['x']
            y = data['y']
            
            # x shape: (T, C, H, W)
            # y shape: (1, H, W)
            
            if self.transform:
                x, y = self.transform(x, y)
                
            return x, y
            
        except Exception as e:
            print(f"❌ Error cargando {file_path}: {e}")
            # Retornar tensores vacíos o ceros para no romper el DataLoader
            # Asumiendo dimensiones estándar 224x224 y contexto 3
            return torch.zeros(3, 12, 224, 224), torch.zeros(1, 224, 224)
