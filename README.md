ActionLSTM – Human Action Recognition from 2D Skeletons (UCF101)

Este proyecto implementa un modelo de deep learning para la clasificación de acciones humanas utilizando el dataset UCF101 Skeleton 2D, basado en coordenadas de 17 puntos clave (keypoints) por frame.
El sistema desarrollado incluye:

Preprocesamiento de datos

Modelo ActionLSTM (LSTM bidireccional)

Entrenamiento usando PyTorch

Evaluación del modelo

Script de inferencia para generar predicciones Top-5

📂 Estructura del proyecto
├── dataset/
│   └── ucf101_2d.pkl
├── checkpoints/
│   └── action_lstm_ucf101.pth
├── main.py          # Entrenamiento y evaluación
├── inference.py     # Predicción (inferencias)
└── README.md

🧠 Modelo: ActionLSTM

ActionLSTM es un modelo basado en un LSTM bidireccional, diseñado para capturar dependencias temporales en secuencias de poses humanas.

Características principales:

LSTM bidireccional

2 capas recurrentes

128 unidades ocultas

Dropout = 0.3

Entrada: secuencias de hasta 120 frames, cada frame con 51 features (x, y y score por articulación)

Salida: 101 clases correspondientes al dataset UCF101

📊 Resultados obtenidos

El modelo se entrenó por 20 épocas utilizando el split oficial train1/test1.

Desempeño general:

Accuracy baseline aleatorio (101 clases): ≈ 0.99 %

Mejor accuracy del modelo en test: ≈ 31 %

Accuracy en entrenamiento: ≈ 50 %

Se usaron las 101 clases completas

Ejemplo de predicción (inferencia):

Ground truth: 0
Top-5 predicciones:

clase 77 (30.9 %)

clase 1 (27.5 %)

clase 0 (9.2 %)

clase 17 (7.3 %)

clase 19 (6.8 %)

🚀 Cómo entrenar el modelo

Ejecuta en la terminal:

python main.py


Los pesos del modelo entrenado se guardarán automáticamente en:

checkpoints/action_lstm_ucf101.pth

🔍 Cómo ejecutar inferencias

Ejecuta:

python inference.py


Para probar diferentes videos del conjunto de prueba, modifica el parámetro idx dentro del archivo inference.py.

📦 Dependencias necesarias

Instala las librerías necesarias:

pip install torch numpy tqdm scikit-learn

📌 Posibles mejoras futuras

Uso de Graph Convolutional Networks (GCN)

Transformers temporales

Modelos híbridos CNN + LSTM

Aumento de datos temporal

Fine-tuning por grupos de clases similares
