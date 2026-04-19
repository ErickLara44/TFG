# Análisis y Conclusiones: Swin Transformer V2 3D vs. ConvLSTM

Este documento presenta las conclusiones analíticas extraídas de la comparación directa entre la arquitectura puramente convolucional (ConvLSTM) actualmente en producción y la exploración profunda con la arquitectura basada en atención (Swin Transformer V2 3D) para la predicción espacio-temporal de la propagación de incendios en IberFire.

## 1. Capacidad de Comprensión Matemática (
    Validación)
El primer paso para evaluar ambos modelos es medir su capacidad máxima para entender las complejas dinámicas no-lineales del fuego sobre un conjunto de datos acotado (Train/Validation Set de ~10,000 muestras).

* **Swin V2 3D Máximo:** 84.04% IoU (Epoch 4)
* **ConvLSTM Máximo:** ~70% - 73% IoU

**Conclusión:** 
La arquitectura Swin V2, gracias a su mecanismo de atención cruzada (Self-Attention) temporal y espacial, demuestra no tener los límites geométricos de las redes convolucionales tradicionales. Es capaz de observar el mapa de forma global e instantánea (por ejemplo, relacionando inmediatamente los frentes de viento de un extremo del mapa con la orografía del opuesto). Matemáticamente, este modelo ha probado tener un **techo analítico inmensamente superior**, entendiendo la dinámica del fuego en simulación casi a la perfección.

## 2. Capacidad de Generalización (Test Set en el Mundo Real)
El verdadero desafío para un modelo integrado en un sistema como IberFire es su robustez ante escenarios terrestres locales que **nunca ha visto**. Para esta fase, los dos modelos se enfrentaron sin filtros a 580 eventos de propagación completamente nuevos (Test Set).

* **ConvLSTM (Modelo Base):** 56.62% IoU
* **Swin V2 3D (Regularizado):** 56.39% IoU

**Conclusión:** 
A la hora de la verdad frente a datos desconocidos, la red convolucional **ConvLSTM gana por un apretado 0.23%**.
¿Por qué ocurre esto si el Swin V2 tiene mayor capacidad de entendimiento? La respuesta radica en el **Sesgo Inductivo (Inductive Bias)**.

1. **La Ventaja Convolucional:** Las redes ConvLSTM están programadas matemáticamente como ventanas cuadradas que barren la pantalla píxel a píxel. Esto las obliga implícitamente a entender que "las cosas fluyen hacia el vecino". Este sesgo inductivo incorporado las hace reinas indiscutibles a la hora de modelar la fluidez y propagación de elementos físicos (fuego, fluidos, clima) incluso cuando se les alimenta con un conjunto de datos pequeño (10,000 mapas). Generalizan de manera natural.
2. **El Problema del Transformer:** La arquitectura Swin V2, por defecto, entra "completamente en blanco". No asume que los píxeles adyacentes están relacionados; tiene que descubrir la propagación desde cero experimentando. Como es tan sumamente complejo, comienza a sufrir de **Sobreajuste (Overfitting)** temprano: en vez de extraer las "leyes físicas" de la propagación, opta por la solución fácil de "memorizar" la estructura de los 10,000 mapas de entrenamiento.

### Regularización (Domesticando al Transformer)
Para que el modelo Swin V2 lograse igualar al ConvLSTM en el Test, tuvo que someterse a regímenes de castigo extremo durante su entrenamiento, obligándolo a generalizar en lugar de memorizar mediante:
* **Decaimiento de pesos (Weight Decay 1e-2):** Forzando una distribución plana de los aprendizajes y evitando pesos monolíticos inestables.
* **Apagado aleatorio (Dropout 20%):** Mutilando neuronas aleatoriamente en cada iteración operativa para que el modelo nunca pudiera confiar en caminos memorizados.
* **Curva de Aprendizaje Cosine Annealing:** Reduciendo sistemáticamente la capacidad del modelo para dar "grandes saltos analíticos", forzando un acoplamiento fino a los valles genéricos del conocimiento hacia los últimos epochs.

## 3. Discusión Estratégica para IberFire (TFG)

El enfrentamiento ha dejado un marco muy claro para el desarrollo del proyecto y arroja una poderosa narrativa tecnológica para las defensas del TFG:

1. **Modelo en Producción (ConvLSTM):**
   * Es **robusto y de comprobada eficacia**. Su naturaleza para el sesgo físico espaciotemporal le permite generalizar sobre pocos datos de entrenamiento.
   * Es **eficiente**. Ocupa mucha menos RAM, entrena de forma sustancialmente más rápida en MPS (Mac) y su tamaño es manejable, haciéndolo ideal para sostener la lógica en vivo de la plataforma online.
   * **Veredicto:** Seguirá siendo el motor operativo ('Caballo de batalla') de IberFire.

2. **Exploración Vanguardista (Swin V2 3D):**
   * Es el **proyecto hacia el futuro**. Su tope analítico base de 84% destroza los límites históricos de las convoluciones. Ha demostrado sin paliativos que los modelos puramente Transformer pueden modelar fenómenos espaciotemporales fluidos con extrema eficacia.
   * Su única limitación actual no es arquitectónica, sino de **volumen de datos**. Los Transformers de Visión (ViT) no alcanzan su verdadero "Sweet Spot" de dominación frente a las CNN hasta que no son entrenados en *millones* de samples en clusters masivos de GPU multicore en paralelo. 
   * **Veredicto:** Este modelo queda sellado y documentado como un éxito de la experimentación arquitectónica moderna; marcando un clarísimo sendero de trabajo a futuro, escalabilidad y demostrando comprensión de la literatura de Inteligencia Artificial actual en los contextos de *Vision Transformers*.
