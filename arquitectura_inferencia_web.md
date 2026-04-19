# Arquitectura de la Plataforma: Motor de Inferencia e Interfaz Web

Para garantizar que los modelos matemáticos y neuronales de Iberfire sean útiles en un entorno de contingencia y vigilancia forestal real, el sistema cuenta con un despliegue operativo orquestado en una arquitectura cliente-servidor (Backend + Frontend).

A continuación se detalla la mecánica de inferencia (motor de cálculo) y el diseño de la interfaz interactiva.

---

## 1. Motor de Inferencia y API (Backend)

Todo el grueso computacional reside en un entorno aislado escrito en **Python** empleando el ecosistema asíncrono **FastAPI** (`src/api/main.py`). El paradigma seguido es exponer los modelos predictivos mediante una arquitectura RESTful (endpoints HTTP).

### 1.1 Proceso Escalable y Protecciones Geo-Temporales
Cada vez que el servidor recibe una petición (`POST /predict/ignition` con `Lat, Lon, y Date`), el ciclo de inferencia activa defensas lógicas vitales antes de desperdiciar capacidad de procesamiento en la tarjeta gráfica/CPU:
*   **Validación de Caja Geográfica:** Un filtro algebraico rechaza inmediatamente predicciones sobre coordenadas marinas o lejanas a la zona de estudio, confinando las ejecuciones al rectángulo ibérico válido.
*   **Validación Temporal:** Constata que la base histórica pueda responder. Se evitan llamadas ficticias futuras no respaldadas por satélites o retroactivas pre-2008.

### 1.2 Extracción de Tensores en Tiempo Real
El script `data_fetcher.py` procede a leer los `Datacubes` colosales en tiempo real, extraer un recorte hiper-dimensionado alrededor de las coordenadas solicitadas recabando la profundidad histórica requerida (ej. 3 días) y entregando un Tensor normalizado listo para la red neuronal.

### 1.3 Inferencia Computacional (PyTorch)
La resolución matemática recae en el archivo `inference.py`, el cual brilla en su optimización para despliegues (Producción):
1.  **Lazy Loading:** Para no saturar la memoria RAM del servidor web en momentos muertos, el peso del modelo `.pth` (*RobustFireIgnitionModel*) solo se inyecta en memoria cuando es invocado bajo demanda por un usuario por primera vez.
2.  **Modo de Evaluación sin Propagación (`No Grad`):** A nivel algorítmico, y como el modelo ya está estrictamente entrenado, se inyectan los tensores cerrando los ramificados tensores automáticos retroactivos mendiante `torch.no_grad()`. Esto permite resolver las inferencias ahorrando más de un 50% de memoria activa en placa y reduciendo la latencia de procesado.
3.  **Clasificación Condicional y Capa Humana:** Las redes neuronales emiten "Logits" puramente matemáticos. La API absorbe este logit con una función *Sigmoide* escalándolo entre `[0, 1]`. Para dotar a las autoridades y prevención humana de decisiones claras, se empaqueta la salida en estratos de emergencia discretos en base a esa probabilidad matemática:
    *   **Bajo:** $< 20\%$
    *   **Moderado:** $20\% \le x < 50\%$
    *   **Alto:** $50\% \le x < 80\%$
    *   **Extremo:** $\ge 80\%$

---

## 2. Aplicación y Visualización Web (Frontend)

Con el fin de democratizar el uso de los complejos modelos algorítmicos, el sistema cuenta con un panel de control interactivo para usuario final, aislado en su propio entorno Web de Alto Rendimiento (`src/web`).

### 2.1 Pila Tecnológica del Cliente
*   **Next.js 16 y React 19:** Actúa como el armazón de la experiencia de usuario (SPA), garantizando visualización dinámica fluida e instanciación de red (fetching) óptimas.
*   **TailwindCSS y Lucide:** Las gráficas y la estilística se ejecutan compitiendo por ser las de menor latencia. Los íconos y las estéticas usan clases utilitarias directas, sin cargar pesados sistemas de CSS convencionales.
*   **Protocolo de Comunicación:** Los conectores HTTP de `Axios` logran una transcodificación limpia de la petición desde la interfaz de React hasta la API en Python sin pérdida de asincronía (el panel web nunca "se cuelga" esperando el modelo matemático).

### 2.2 Subsistema Geoespacial (Leaflet GIS)
El núcleo de la experiencia Web reside en la librería de cartografía pura de código abierto `Leaflet` (implementada reactivamente mediante `react-leaflet`). 

Esto le brinda a las autoridades de prevención una ventaja inmensa: se elimina cualquier complejidad y fricción técnica. Sobre este lienzo interactivo mundial satelital, los agentes simplemente se desplazan a su jurisdicción, hacen click en terrenos vulnerables y en cuestión de segundos, la web invoca todo el tubo del backend (Data Extraction $\to$ Inferencia GPU $\to$ Conversión Discreta) y despliega gráficos ilustrativos en pantalla.
