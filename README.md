# Reconhecimento de Dígitos com Redes Neurais

Este projeto utiliza uma rede neural para prever números escritos à mão com base no dataset **NumberWritten**. O código implementa um modelo de aprendizado profundo para classificação de imagens de dígitos. Abaixo, temos uma descrição detalhada do dataset e do funcionamento do código.

## Sobre o Dataset: NumberWritten
O dataset **NumberWritten** é composto por cada pixels de uma imagem de números manuscritos representadas em uma matriz de 8x8 pixels. Cada imagem é convertida em uma sequência de 64 atributos que descrevem os níveis de intensidade de cinza de cada pixel. A classificação associada indica o número correspondente (0 a 9).

O dataset, com redução de dimensionabilidade, é projetado para ser compacto e eficiente, permitindo um aprendizado rápido e eficaz. Sua simplicidade o torna ideal para treinar e testar modelos de classificação de imagens em redes neurais.

Neste projeto, o dataset é dividido em três partes:
- **Treinamento:** Para ajustar os pesos do modelo.
- **Validação:** Para avaliar a precisão do modelo durante o treinamento.
- **Teste:** Para medir a performance final do modelo com dados nunca antes vistos.

## Estrutura do Código
Abaixo, destacamos as principais etapas:

### 1. Importação de Bibliotecas
```python
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```
As bibliotecas são usadas para manipulação de dados, visualização, construção e avaliação do modelo.

### 2. Carregamento e Exploração do Dataset
- O dataset é carregado a partir de um arquivo CSV.
- A distribuição das classes é analisada para garantir balanceamento.

```python
df = pd.read_csv("NumbersWritten.csv")
print(df["Classification"].value_counts())
```

### 3. Visualização de Imagens
As imagens dos dígitos são reconstruídas e exibidas para verificação visual.

```python
imagens = df.iloc[:, :-1].to_numpy().reshape(-1, 8, 8)
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(imagens[i], cmap="gray")
    plt.title(f"Label: {i}")
    plt.axis("off")
plt.tight_layout()
plt.show()
```

### 4. Divisão do Dataset
O dataset é dividido em:
- Dados de treinamento e validação.
- Dados de teste separados para previsões futuras.

```python
X = df.drop("Classification", axis=1)
y = df["Classification"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_classfication = to_categorical(y_train, num_classes = 10)
y_val_classfication = to_categorical(y_val, num_classes = 10)

X_test = df_previsao.drop("Classification", axis = 1)
y_test = df_previsao["Classification"]
```

### 5. Construção do Modelo
O modelo é uma rede neural sequencial com camadas densas. Ele utiliza a função de ativação ReLU para camadas ocultas e softmax para a camada de saída.

```python
model = Sequential([
    Input(shape=(64,)),
    Dense(10, activation="linear"),
    Dense(16, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
```

### 6. Treinamento do Modelo
O modelo é treinado por 100 épocas, com validação em cada época.

```python
history = model.fit(
    X_train, to_categorical(y_train, num_classes=10),
    validation_data = (X_val, y_val_classfication),
    epochs=100,
    verbose=1
)
```

### 7. Avaliação e Visualização dos Resultados
Gráficos de perda e precisão são gerados para monitorar o desempenho do modelo.

```python
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.show()
```

### 8. Previsão e Métricas
O modelo é avaliado com dados de teste. A precisão e o relatório de classificação são exibidos.

```python
y_pred = np.argmax(model.predict(X_test), axis=1)
print("Acc:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Conclusão
Este projeto demonstra como construir e treinar uma rede neural para reconhecimento de dígitos manuscritos. O dataset **NumberWritten** é simples, mas eficaz para entender os conceitos básicos de classificação de imagens com redes neurais. O código é modular e pode ser facilmente adaptado para outros datasets ou aplicações semelhantes.
