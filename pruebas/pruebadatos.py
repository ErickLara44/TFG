# ============================================================
# 🌍 Menú interactivo IberFire + EFFIS
# Crea la cuadrícula IberFire, filtra incendios y genera etiquetas
# ============================================================

import geopandas as gpd
import shapely
from shapely.geometry import Polygon
import numpy as np
import os
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Función auxiliar: Cargar límites de España y CCAA
# ------------------------------------------------------------
def cargar_limites_espana():
    """Carga los límites de España (Península y Baleares, sin Canarias)"""
    try:
        # Intentar cargar desde GADM (base de datos global de límites administrativos)
        print("🌍 Descargando límites administrativos de España...")
        url_spain = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ESP_1.json"
        import requests
        response = requests.get(url_spain, timeout=30)
        
        if response.status_code == 200:
            import json
            from io import StringIO
            ccaa = gpd.read_file(StringIO(response.text))
            ccaa = ccaa.to_crs(epsg=3035)
            
            # Filtrar para excluir Canarias (y opcionalmente Ceuta y Melilla)
            excluir = ['Canarias', 'Islas Canarias', 'Ceuta', 'Melilla']
            if 'NAME_1' in ccaa.columns:
                ccaa = ccaa[~ccaa['NAME_1'].isin(excluir)].copy()
            
            print(f"✅ Límites cargados: {len(ccaa)} comunidades (Península y Baleares)")
            return ccaa
        else:
            print("⚠️ No se pudieron descargar los límites")
            return None
    except Exception as e:
        print(f"⚠️ Error al cargar límites: {e}")
        return None

# ------------------------------------------------------------
# Función 1️⃣: Crear cuadrícula IberFire (1 km × 1 km)
# ------------------------------------------------------------
def crear_grid():
    print("\n🧱 Creando cuadrícula IberFire (1x1 km)...")

    # Área aproximada de cobertura (España, EPSG:3035)
    xmin, ymin = 2400000, 1400000
    xmax, ymax = 4400000, 3200000
    cell_size = 1000  # 1 km

    cols = np.arange(xmin, xmax, cell_size)
    rows = np.arange(ymin, ymax, cell_size)

    polygons, ids = [], []
    counter = 0

    for x in cols:
        for y in rows:
            counter += 1
            polygons.append(Polygon([
                (x, y),
                (x + cell_size, y),
                (x + cell_size, y + cell_size),
                (x, y + cell_size)
            ]))
            ids.append(counter)

    grid = gpd.GeoDataFrame({"cell_id": ids}, geometry=polygons, crs="EPSG:3035")
    grid.to_file("iberfire_grid_1km.geojson", driver="GeoJSON")

    print(f"✅ Cuadrícula creada con {len(grid)} celdas.")
    print("📁 Guardada como 'iberfire_grid_1km.geojson'")

# ------------------------------------------------------------
# Función 2️⃣: Filtrar incendios EFFIS (>30 ha y columnas útiles)
# ------------------------------------------------------------
def filtrar_effis():
    print("\n🔥 Filtrando incendios EFFIS (>30 ha)...")

    file_path = "/Users/erickmollinedolara/Erick/Uni/TFG/EFJ/fe.json"
    if not os.path.exists(file_path):
        print("❌ Archivo no encontrado.")
        return

    fires = gpd.read_file(file_path)
    if "area_ha" not in fires.columns:
        print("❌ No se encontró la columna 'area_ha'. Revisa tu archivo.")
        return

    fires_big = fires[fires["area_ha"] > 30].copy()

    # Mantener solo columnas útiles
    keep_cols = [c for c in ["area_ha", "geometry"] if c in fires_big.columns]
    fires_big = fires_big[keep_cols]

    fires_big = fires_big.to_crs(epsg=3035)
    
    # Cargar límites y filtrar incendios solo en península y Baleares
    ccaa = cargar_limites_espana()
    if ccaa is not None:
        peninsula_baleares = ccaa.union_all()
        fires_big = fires_big[fires_big.intersects(peninsula_baleares)].copy()
        print(f"✅ Incendios filtrados en Península y Baleares: {len(fires_big)}")
    
    fires_big.to_file("effis_fires_mayor30ha.geojson", driver="GeoJSON")
    print("📁 Guardado como 'effis_fires_mayor30ha.geojson'")

    # --- 📊 VISUALIZACIÓN DE INCENDIOS FILTRADOS ---
    print("\n📊 Generando visualización...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Fondo del mapa (comunidades autónomas sin Canarias)
    if ccaa is not None:
        ccaa.plot(
            ax=ax,
            facecolor='lightgray',
            edgecolor='black',
            linewidth=1.2,
            alpha=0.3
        )
        
        # Añadir nombres de las comunidades autónomas
        for idx, row in ccaa.iterrows():
            centroid = row.geometry.centroid
            if 'NAME_1' in ccaa.columns:
                ax.text(centroid.x, centroid.y, row['NAME_1'], 
                       fontsize=7, ha='center', style='italic', alpha=0.6)
    
    # Plot de los incendios coloreados por área
    fires_big.plot(
        ax=ax,
        column="area_ha",
        cmap="YlOrRd",
        legend=True,
        legend_kwds={
            'label': "Área (ha)",
            'orientation': "vertical",
            'shrink': 0.7
        },
        edgecolor='darkred',
        linewidth=0.5,
        alpha=0.8
    )
    
    ax.set_title(
        f"🔥 Incendios EFFIS > 30 ha (Península y Baleares)\n(Total: {len(fires_big)} incendios)",
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel("X (EPSG:3035)", fontsize=10)
    ax.set_ylabel("Y (EPSG:3035)", fontsize=10)
    
    # Añadir estadísticas en el plot
    stats_text = f"Área total: {fires_big['area_ha'].sum():.0f} ha\n"
    stats_text += f"Área media: {fires_big['area_ha'].mean():.0f} ha\n"
    stats_text += f"Área máxima: {fires_big['area_ha'].max():.0f} ha"
    
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Ajustar límites del mapa a península y Baleares
    if ccaa is not None:
        ax.set_xlim(ccaa.total_bounds[0] - 50000, ccaa.total_bounds[2] + 50000)
        ax.set_ylim(ccaa.total_bounds[1] - 50000, ccaa.total_bounds[3] + 50000)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualización completada")

# ------------------------------------------------------------
# Función 3️⃣: Intersecar incendios con cuadrícula IberFire
# ------------------------------------------------------------
def comparar_y_etiquetar():
    print("\n🔍 Generando etiquetas 'is_fire' (Península y Baleares, 2018-2024, >30 ha)...")

    grid = gpd.read_file("iberfire_grid_1km.geojson")
    fires_big = gpd.read_file("effis_fires_mayor30ha.geojson")

    # Cargar límites de comunidades autónomas (sin Canarias)
    ccaa = cargar_limites_espana()
    
    if ccaa is not None:
        # Filtrar solo las celdas que intersectan con Península y Baleares
        peninsula_baleares = ccaa.union_all()
        grid = grid[grid.intersects(peninsula_baleares)].copy()
        print(f"✅ Cuadrícula recortada a Península y Baleares: {len(grid)} celdas")

    # Unir geometrías y hacer intersección
    grid["is_fire"] = grid.intersects(fires_big.union_all()).astype(int)

    # Guardar resultado
    grid.to_file("is_fire_spain_30ha_2018_2024.geojson", driver="GeoJSON")
    print(f"✅ Etiquetas creadas: {grid['is_fire'].sum()} celdas con fuego.")
    print("📁 Guardado como 'is_fire_spain_30ha_2018_2024.geojson'")

    # --- Visualización ---
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Fondo con límites de comunidades autónomas (sin Canarias)
    if ccaa is not None:
        ccaa.plot(
            ax=ax,
            facecolor='white',
            edgecolor='black',
            linewidth=1.5,
            alpha=1
        )
        
        # Añadir nombres de las comunidades autónomas
        for idx, row in ccaa.iterrows():
            centroid = row.geometry.centroid
            if 'NAME_1' in ccaa.columns:
                ax.text(centroid.x, centroid.y, row['NAME_1'], 
                       fontsize=7, ha='center', style='italic', 
                       alpha=0.5, color='gray')
    
    # Solo plotear las celdas CON fuego (is_fire == 1)
    grid_con_fuego = grid[grid["is_fire"] == 1]
    grid_con_fuego.plot(
        ax=ax,
        color='red',
        alpha=0.6,
        edgecolor='darkred',
        linewidth=0.2
    )
    
    # Contornos de los incendios originales
    fires_big.plot(
        ax=ax,
        facecolor='none',
        edgecolor='black',
        linewidth=0.8
    )
    
    ax.set_title(
        f"🧱 Celdas con incendios EFFIS (Península y Baleares, >30 ha, 2018–2024)\n{grid_con_fuego.shape[0]} celdas afectadas de {len(grid)} totales",
        fontsize=12,
        fontweight='bold'
    )
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 🧭 Menú principal
# ------------------------------------------------------------
def menu():
    while True:
        print("\n===========================================")
        print("🔥 MENÚ IBERFIRE – EFFIS")
        print("===========================================")
        print("1️⃣  Crear cuadrícula IberFire (1 km²)")
        print("2️⃣  Filtrar incendios EFFIS (>30 ha)")
        print("3️⃣  Intersectar cuadrícula con incendios")
        print("0️⃣  Salir")
        print("===========================================")

        opcion = input("👉 Elige una opción: ").strip()

        if opcion == "1":
            crear_grid()
        elif opcion == "2":
            filtrar_effis()
        elif opcion == "3":
            comparar_y_etiquetar()
        elif opcion == "0":
            print("👋 Saliendo del programa...")
            break
        else:
            print("❌ Opción no válida. Intenta de nuevo.")

# ------------------------------------------------------------
# Ejecutar menú
# ------------------------------------------------------------
if __name__ == "__main__":
    menu()