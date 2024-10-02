#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:22:20 2024

@author: bettu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

# Carregar Dataset
diabetes = load_diabetes()
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print("Dados carregados com sucesso.")

# Estandarizar Dados
scaler = StandardScaler()
# The standard score of a sample `x` is calculated as:
# z = (x - u) / s
data_scaled = scaler.fit_transform(data)
print("Dados estandarizados com sucesso.")

# Calcular Matriz de Correlação
corr_matrix = np.corrcoef(data_scaled, rowvar=False)  # Return Pearson product-moment correlation coefficients.
print("Matriz de correlação calculada.")
print(corr_matrix)

# Plotar a Matriz de Correlação
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=data.columns, yticklabels=data.columns)
plt.title('Matriz de Correlação')
plt.show()

# Testes de Adequação dos Dados
# Teste de esfericidade de Bartlett
bartlett_test, p_value = calculate_bartlett_sphericity(data_scaled)
print(f"Teste de Esfericidade de Bartlett: {bartlett_test}, p-valor: {p_value}")

# Medida de Adequação de Kaiser-Meyer-Olkin (KMO)
kmo_all, kmo_model = calculate_kmo(data_scaled)
print(f"KMO: {kmo_model}")

# Aplicar Análise Fatorial Exploratória (EFA)
n_factors = 3  # Número de fatores
fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
fa.fit(data_scaled)

# Cargas Fatoriais
loadings = fa.loadings_

# Variância Explicada
variance = fa.get_factor_variance()

# Plotar Scree Plot
ev, v = np.linalg.eig(corr_matrix)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ev) + 1), ev, marker='o')
plt.xlabel('Número de Fatores')
plt.ylabel('Autovalores')
plt.title('Scree Plot')
plt.show()

# Heatmap da Matriz de Cargas Fatoriais
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(loadings, columns=[f'Fator {i+1}' for i in range(n_factors)]), annot=True, cmap='coolwarm')
plt.title('Matriz de Cargas Fatoriais')
plt.show()

# Biplot dos Fatores 1 e 2
plt.figure(figsize=(8, 6))
plt.scatter(loadings[:, 0], loadings[:, 1])
for i, txt in enumerate(data.columns):
    plt.annotate(txt, (loadings[i, 0], loadings[i, 1]), textcoords="offset points", xytext=(5,-5))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('Fator 1')
plt.ylabel('Fator 2')
plt.title('Biplot dos Fatores 1 e 2')
plt.grid()
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.show()

# Biplot dos Fatores 2 e 3
plt.figure(figsize=(8, 6))
plt.scatter(loadings[:, 1], loadings[:, 2])
for i, txt in enumerate(data.columns):
    plt.annotate(txt, (loadings[i, 1], loadings[i, 2]), textcoords="offset points", xytext=(5,-5))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('Fator 2')
plt.ylabel('Fator 3')
plt.title('Biplot dos Fatores 2 e 3')
plt.grid()
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.show()

# Salvar Relatório
with open("relatorio_analise_fatorial_exploratoria.txt", "w") as file:
    file.write("Cargas Fatoriais:\n")
    file.write(pd.DataFrame(loadings, columns=[f'Fator {i+1}' for i in range(n_factors)]).to_string())
    file.write("\n\nVariância Explicada:\n")
    file.write(pd.DataFrame(variance, index=['Variância', 'Proporção', 'Proporção Acumulada'], columns=[f'Fator {i+1}' for i in range(n_factors)]).to_string())

print("Relatório salvo como relatorio_analise_fatorial_exploratoria.txt.")
