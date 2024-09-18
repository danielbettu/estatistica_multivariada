#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:48:32 2024

@author: bettu
"""

from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data

# Aplicar agrupamento hierárquico
Z = linkage(X, 'ward')

# Visualizar o dendrograma
plt.figure(figsize=(50, 31))
dendrogram(Z)
plt.title('Dendrograma do Agrupamento Hierárquico')
plt.xlabel('Índice da Amostra')
plt.ylabel('Distância')
plt.show()
