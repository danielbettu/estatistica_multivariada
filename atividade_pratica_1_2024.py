#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 19:10:08 2024

@author: bettu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Carregar dados do arquivo de entrada em txt
dataset = pd.read_csv('https://raw.githubusercontent.com/danielbettu/estatistica_multivariada/main/sample_data_biased.csv', sep= ",", header=0)

# Definição de funções
# Função para calcular os quartis e a mediana
def calculate_statistics(data):
    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    return q1, median, q3

# Definir uma função para classificar as cores com base no valor da porosidade
def classify_color_perm_porosity(porosity):
    if porosity <= q1_porosity:
        return 'blue'
    elif porosity <= median_porosity:
        return 'green'
    elif porosity <= q3_porosity:
        return 'orange'
    else:
        return 'red'
    
def classify_color_perm(perm):
    if perm <= q1_perm:
        return 'blue'
    elif perm <= median_perm:
        return 'green'
    elif perm <= q3_perm:
        return 'orange'
    else:
        return 'red'

# Plotar histograma de Porosity
counts, bins, patches = plt.hist(dataset['Porosity'], bins=10, edgecolor='black')
plt.xlabel('Porosity')
plt.ylabel('Frequência')
plt.xticks(bins, fontsize=8)  # Ajustar os valores do eixo x
# plt.title('Histograma de Dados Aleatórios')
plt.show()

# Plotar histograma de Perm
counts, bins, patches = plt.hist(dataset['Perm'], bins=10, edgecolor='black')
plt.xlabel('Perm')
plt.ylabel('Frequência')
plt.xticks(bins, fontsize=8)  # Ajustar os valores do eixo x
# plt.title('Histograma de Dados Aleatórios')
plt.show()

# Calcular os quartis e a mediana para Porosity
q1_porosity, median_porosity, q3_porosity = calculate_statistics(dataset['Porosity'])
print(f"Porosity - 1º Quartil: {q1_porosity}, Mediana: {median_porosity}, 3º Quartil: {q3_porosity}")

# Calcular os quartis e a mediana para Perm
q1_perm, median_perm, q3_perm = calculate_statistics(dataset['Perm'])
print(f"Perm - 1º Quartil: {q1_perm}, Mediana: {median_perm}, 3º Quartil: {q3_perm}")

# Plotar boxplot para Porosity
plt.boxplot(dataset['Porosity'])
plt.xlabel('Porosity')
plt.ylabel('Valores')
plt.title('Boxplot de Porosity')
plt.show()

# Plotar boxplot para Perm
plt.boxplot(dataset['Perm'])
plt.xlabel('Perm')
plt.ylabel('Valores')
plt.title('Boxplot de Perm')
plt.show()

# Aplicar transformação logarítmica aos dados de Perm
log_perm = np.log(dataset['Perm'])
q1_log_perm = np.log(q1_perm)
q3_log_perm = np.log(q3_perm)
median_log_perm = np.log(median_perm)

# Plotar boxplot para log_perm
plt.boxplot(log_perm)
plt.xlabel('Log_Perm')
plt.ylabel('Valores')
plt.title('Boxplot de Log_Perm')
plt.show()

# Aplicar a função de classificação de cores à coluna 'Porosity'
colors = dataset['Porosity'].apply(classify_color_perm_porosity)

# Plotar o diagrama de dispersão
plt.scatter(dataset['X'], dataset['Y'], c=colors)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Diagrama de Dispersão entre X e Y Classificado por Porosidade')
plt.show()

# Aplicar a função de classificação de cores à coluna 'Perm'
colors = dataset['Perm'].apply(classify_color_perm)

# Plotar o diagrama de dispersão
plt.scatter(dataset['X'], dataset['Y'], c=colors)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Diagrama de Dispersão entre X e Y Classificado por Perm')
plt.show()

# Plotar o diagrama de dispersão entre 'Porosity' e 'Perm'
plt.scatter(dataset['Porosity'], dataset['Perm'])
plt.xlabel('Porosity')
plt.ylabel('Perm')
plt.title('Diagrama de Dispersão entre Porosity e Perm')
plt.show()

# Plotar o diagrama de dispersão entre 'Porosity' e 'log_Perm'
plt.scatter(dataset['Porosity'], log_perm)
plt.xlabel('Porosity')
plt.ylabel('Log_Perm')
plt.title('Diagrama de Dispersão entre Porosity e Log_Perm')
plt.show()

# Calcular o logaritmo de 'Perm'
dataset['log_Perm'] = np.log(dataset['Perm'])

# Definir as variáveis independentes (X) e dependentes (y)
X = dataset['Porosity']
y = dataset['log_Perm']

# Realizar a regressão linear usando scipy.stats.linregress
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

# Fazer previsões
y_pred = intercept + slope * X

# Calcular o coeficiente de determinação (R²)
r2 = r_value**2

# Plotar o diagrama de dispersão entre 'Porosity' e 'log_Perm'
plt.scatter(X, y, label='Dados')
plt.plot(X, y_pred, color='red', label='Regressão Linear')
plt.xlabel('Porosity')
plt.ylabel('Log_Perm')
plt.title(f'Diagrama de Dispersão entre Porosity e Log_Perm\nCoeficiente de Determinação (R²): {r2:.2f}')
plt.legend()
plt.show()

# Salvar o coeficiente de determinação em uma variável
coeficiente_determinacao = r2

print(f"Coeficiente de Determinação (R²): {coeficiente_determinacao:.2f}")