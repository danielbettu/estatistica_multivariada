#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:55:21 2024

@author: bettu
"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data

# Aplicar PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

# Visualização dos dados após PCA
combinacoes = [(0, 1), (1, 2), (2, 3)]

for i, (x, y) in enumerate(combinacoes):
    plt.figure(i)
    plt.scatter(X_pca[:, x], X_pca[:, y])
    plt.xlabel(f'Componente Principal {x + 1}')
    plt.ylabel(f'Componente Principal {y + 1}')
    plt.title(f'Visualização dos Dados Após PCA - Componentes {x + 1} e {y + 1}')
    plt.show()

# Scores dos componentes principais
pc1_scores = X_pca[:, 0]
pc2_scores = X_pca[:, 1]

# Cargas (loadings) das variáveis originais
loadings = pca.components_.T

# Plotar o biplot para todas as combinações de componentes principais
combinacoes_biplot = [(0, 1), (1, 2), (2, 3)]

for i, (x, y) in enumerate(combinacoes_biplot):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, x], X_pca[:, y], alpha=0.5)
    plt.xlabel(f'Componente Principal {x + 1}')
    plt.ylabel(f'Componente Principal {y + 1}')
    plt.title(f'Biplot PCA - Componentes {x + 1} e {y + 1}')
    
    # Adicionar vetores de carga
    for k, var in enumerate(['Sepal length', 'Sepal width', 'Petal length', 'Petal width']):
        plt.arrow(0, 0, loadings[k, x], loadings[k, y], color='r', alpha=0.5)
        plt.text(loadings[k, x] * 1.15, loadings[k, y] * 1.15, var, color='g', ha='center', va='center')
    
    plt.grid()
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