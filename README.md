# ActionLSTM – Human Action Recognition from 2D Skeletons (UCF101)

Este proyecto implementa un modelo de deep learning para la clasificación de acciones humanas utilizando el dataset **UCF101 Skeleton 2D**, basado en coordenadas de 17 puntos clave (keypoints) por frame.

El sistema desarrollado incluye:
- Preprocesamiento de keypoints 2D
- Modelo **ActionLSTM** (LSTM bidireccional)
- Modelo **ActionMLP** (temporal pooling + MLP)
- Entrenamiento usando PyTorch
- Evaluación cuantitativa del modelo
- Script de inferencia con nombres de clase para generar predicciones Top-5 

---

## 📂 Estructura del proyecto

dataset/
└── ucf101_2d.pkl

checkpoints/
└── action_lstm_ucf101.pth
└── action_mlp_ucf101.pth

main.py   # Entrenamiento y evaluación

inference.py   # Predicción (inferencias)

README.md

---

## 🧠 Modelos Implementados

### 🔹 1) **ActionLSTM (modelo principal)**   

ActionLSTM es un modelo basado en un LSTM bidireccional, diseñado para capturar dependencias temporales en secuencias de poses humanas.

**Características:**
- LSTM bidireccional  
- 2 capas recurrentes  
- 128 unidades ocultas  
- Dropout = 0.3  
- Entrada: secuencias de hasta 120 frames, cada frame con 51 features (x, y y score por articulación)  
- Salida: 101 clases correspondientes al dataset UCF101  


### 🔹 2) **ActionMLP (baseline adicional)**  

Implementado para comparar arquitecturas del proyecto.

**Características:**
- Pooling temporal sobre todos los frames válidos  
- MLP de dos capas  
- Dropout = 0.3  
- Rápido y eficiente, pero sin modelar temporalidad  

Sirve para ver claramente las ventajas del LSTM.

---


## 📊 Resultados obtenidos

Los dos modelos se entrenaron durante 20 épocas usando el split oficial `train1/test1`.

### **Resumen de métricas**

| Modelo         | Arquitectura              | Accuracy Test | Accuracy Train |
|----------------|---------------------------|----------------|----------------|
| Aleatorio      | Predicción uniforme       | 0.99%         | —              |
| **ActionMLP**  | MLP con pooling temporal  | **28.1%**     | 31%            |
| **ActionLSTM** | LSTM bidireccional        | **31%**       | 50%            |

**Conclusiones:**
- Ambos modelos superan ampliamente al baseline aleatorio.  
- El LSTM obtiene mejor desempeño al capturar dependencias temporales.  
- El MLP ofrece una comparación sólida y valida experimentalmente la elección del LSTM como arquitectura principal.  

---

## 🔍 Inferencia con nombres de clase

El script `inference.py` muestra predicciones reales con nombres de clase:

---

## 🚀 Cómo entrenar el modelo

Ejecuta en la terminal:

`python main.py`

Para elegir el modelo se tendrá que cambiar el parámetro `MODEL_TYPE` dentro de `main.py`.

Los pesos del modelo entrenado se guardarán dependiendo del modelo utilizado:

`checkpoints/action_lstm_ucf101.pth`
`checkpoints/action_mlp_ucf101.pth`

---

## 🔍 Cómo ejecutar inferencias

Ejecuta:

 `python inference.py`

- Para probar diferentes videos del conjunto de prueba, cambia el parámetro `idx` dentro de `inference.py`.
- Para elegir el modelo que utilizará `inference.py` se tendrá que cambiar el parámetro `MODEL_TYPE` dentro de `main.py`.

---

## 📦 Dependencias necesarias

Instala las librerías requeridas:

 `pip install torch numpy tqdm scikit-learn`

---

## 📌 Posibles mejoras futuras

- Usar Graph Convolutional Networks (GCN) para modelar relaciones entre articulaciones.
- Implementar Transformers temporales (TimeSformer, PoseFormer).
- Modelos híbridos CNN + LSTM
- Aplicar data augmentation temporal (jittering, frame dropping, scaling).
- Ajustar hiperparámetros y realizar fine-tuning específico por categoría.
- Entrenar modelos pre-entrenados en esqueletos como ST-GCN.

---

## 📄 Licencia

Proyecto desarrollado con fines académicos dentro del módulo de Deep Learning.





