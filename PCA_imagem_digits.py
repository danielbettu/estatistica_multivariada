#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 08:47:10 2024

@author: bettu

Carrega o dataset de dígitos manuscritos (8x8 pixels) do sklearn.
Seleciona a primeira imagem do dataset e a exibe.
Aplica o PCA para reduzir a dimensionalidade da imagem.
Reconstrói a imagem usando apenas os 16 componentes principais.
Exibe a imagem reconstruída lado a lado com a original para comparação.
 
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np

# Carregar o dataset de dígitos
digits = load_digits()

# Selecionar uma imagem do dataset
image = digits.images[0]  # primeira imagem
original_shape = image.shape

# Redimensionar todas as imagens do dataset para aplicar o PCA
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))  # flatten todas as imagens

# Aplicar PCA com 16 componentes ao dataset completo
pca = PCA(n_components=16)
pca.fit(data)

# Transformar a imagem selecionada para os componentes principais
image_pca = pca.transform(image.flatten().reshape(1, -1))

# Reconstruir a imagem a partir dos componentes principais
image_reconstructed = pca.inverse_transform(image_pca)
image_reconstructed = image_reconstructed.reshape(original_shape)

# Aplicar PCA ao dataset completo, com 2 componentes principais
pca_2 = PCA(n_components=2)
data_pca = pca_2.fit_transform(data)

# Plotar todas as imagens reconstruídas com os componentes principais
fig, axs = plt.subplots(4, 4, figsize=(10, 10))

for i in range(16):
    pca_n = PCA(n_components=i+1)
    image_pca_n = pca_n.fit_transform(data)  # Ajustar e transformar o dataset com n componentes
    image_reconstructed_n = pca_n.inverse_transform(image_pca_n[0])  # Reconstruir a primeira imagem
    image_reconstructed_n = image_reconstructed_n.reshape(original_shape)  # Redimensionar para a forma original

    ax = axs[i // 4, i % 4]
    img = ax.imshow(image_reconstructed_n, cmap='gray')
    ax.set_title(f'{i+1} componentes')
    ax.axis('off')

    # Adicionar a escala de cores (colorbar) para cada imagem
    fig.colorbar(img, ax=ax)

plt.suptitle('Imagens Reconstruídas com Diferentes Componentes PCA')
plt.tight_layout()
plt.show()

# Plotar a imagem original ao lado das reconstruídas
plt.figure(figsize=(6, 3))

# Mostrar a imagem original
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Imagem Original")
plt.axis('off')

# Mostrar a última imagem reconstruída (16 componentes)
plt.subplot(1, 2, 2)
plt.imshow(image_reconstructed, cmap='gray')
plt.title(f"Imagem Reconstruída (16 componentes)")
plt.axis('off')

plt.show()

# Criar biplots das duas primeiras componentes principais
plt.figure(figsize=(8, 6))

# Plotar os pontos dos dois primeiros componentes
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=digits.target, cmap='Spectral', s=50, edgecolor='k')
plt.colorbar()
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Biplot PCA - Digits Dataset')

# Calcular os "loadings" (carregamentos) para as variáveis originais
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Adicionar vetores de loadings ao gráfico para mostrar a direção das variáveis originais
for i, feature in enumerate(loadings):
    plt.arrow(0, 0, feature[0], feature[1], color='r', alpha=0.5, head_width=0.05)
    plt.text(feature[0] * 1.2, feature[1] * 1.2, str(i), color='r', ha='center', va='center')

plt.grid(True)
plt.show()

# Calcular a variância explicada acumulada
explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)

# Plotar a variância explicada acumulada
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio_cumulative) + 1), explained_variance_ratio_cumulative, marker='o', linestyle='--')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada')
plt.title('Variância Explicada Acumulada por PCA')
plt.grid(True)
plt.show()

# Convertendo os resultados para um dicionário
pca_result = {
    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
    'singular_values': pca.singular_values_.tolist(),
    'components': pca.components_.tolist(),
    'mean': pca.mean_.tolist()
}

# Exibindo o dicionário
print(pca_result)