import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def main():
    npz_path = "outputs_spread/global_grid_data.npz"
    if not os.path.exists(npz_path):
        print(f"❌ Error: No se encuentra {npz_path}.")
        print("💡 Debes ejecutar primero: python scripts/visualize_global_grid_errors.py para generar las matrices.")
        return
        
    print(f"📦 Cargando datos pre-calculados desde {npz_path}...")
    data = np.load(npz_path)
    grid_T0 = data['grid_T0']
    grid_TP = data['grid_TP']
    grid_FN = data['grid_FN']
    grid_FP = data['grid_FP']
    spain_mask = data['spain_mask']
    
    H_full, W_full = grid_T0.shape
    
    print("🎨 Abriendo visor interactivo de Matplotlib...")
    print("👉 Usa la HERRAMIENTA LUPA de la barra inferior para hacer ZOOM en la zona que quieras.")
    print("👉 Usa las FLECHAS (Pan) para Moverte.")
    print("👉 Cierra la ventana cuando termines.")
    
    # Creamos la figura interactiva (tamaño manejable para monitor)
    fig, ax = plt.subplots(figsize=(15, 12), facecolor='white')
    
    # 1. Fondo de España
    ax.imshow(spain_mask, cmap='gray', alpha=0.3)
    
    # 2. Fuego T0 (Morado)
    m_t0 = np.ma.masked_where(grid_T0 == 0, grid_T0)
    ax.imshow(m_t0, cmap=ListedColormap(['purple']), alpha=0.5, interpolation='none')
    
    # 3. False Positives (Cian/Heatmap)
    grid_FP_norm = grid_FP / (np.max(grid_FP) + 1e-9)
    m_fp = np.ma.masked_where(grid_FP == 0, grid_FP_norm)
    ax.imshow(m_fp, cmap=ListedColormap(['cyan']), alpha=0.6, interpolation='none')
    
    # 4. False Negatives (Rojo)
    m_fn = np.ma.masked_where(grid_FN == 0, grid_FN)
    ax.imshow(m_fn, cmap=ListedColormap(['red']), alpha=0.9, interpolation='none')
    
    # 5. True Positives (Verde Lima)
    m_tp = np.ma.masked_where(grid_TP == 0, grid_TP)
    ax.imshow(m_tp, cmap=ListedColormap(['lime']), alpha=1.0, interpolation='none')
    
    # DIBUJAR LA CUADRÍCULA (GRID)
    ax.set_xticks(np.arange(-.5, W_full, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H_full, 1), minor=True)
    
    # En pantalla interactiva, un grosor de 0.5 o 1 es lo ideal para ver las celdas
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.6)
    
    # Ocultar los ticks (los numeritos)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # Leyenda
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    custom_lines = [
        Line2D([0], [0], color='purple', lw=4, alpha=0.5, label='Fuego t=0 (Histórico)'),
        Line2D([0], [0], color='lime', lw=4, label='TP: Acierto (Prob > 50%)'),
        Line2D([0], [0], color='red', lw=4, label='FN: Omisión (No lo vio)'),
        mpatches.Patch(color='cyan', alpha=0.6, label='FP: Falsa Alarma (Prob > 50%)')
    ]
    ax.legend(handles=custom_lines, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=10)
    
    plt.title("Visor Interactivo de Cuadrícula de Errores (1 Celda = 1km x 1km)", fontsize=14)
    plt.tight_layout()
    
    # Mostrar la ventana interactiva
    plt.show()

if __name__ == "__main__":
    main()
