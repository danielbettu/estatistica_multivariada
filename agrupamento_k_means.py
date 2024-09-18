#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:46:49 2024

@author: bettu
"""
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data

# Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualizar os clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=90, c='red', marker='X', label='Centróides')

# Configurações do gráfico
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.title('Agrupamento K-means no Conjunto de Dados Iris')
plt.legend()
plt.show()

# Calcular o coeficiente de silhueta
silhouette_avg = silhouette_score(X, labels)
print(f'Coeficiente de Silhueta: {silhouette_avg:.2f}')

