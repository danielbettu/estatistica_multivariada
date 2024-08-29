#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:40:07 2024

@author: bettu
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# URL do arquivo CSV
url = "https://raw.githubusercontent.com/danielbettu/estatistica_multivariada/main/matriz_2_1_Landim.csv"

# Carregar o CSV em um DataFrame
dataset = pd.read_csv(url)

# Gerar uma lista com os nomes das colunas
colunas = list(dataset.columns)

# Definir a variável dependente y (primeira coluna)
y = dataset[colunas[0]]
# Definir as variáveis independentes x (todas as demais colunas)
x = dataset[colunas[1:]]

# Criar o modelo de regressão linear
modelo = LinearRegression()

# Treinar o modelo
modelo.fit(x, y)

# Fazer previsões no conjunto de teste
y_pred = modelo.predict(x)

# Avaliar o modelo
mse = mean_squared_error(y, y_pred)
r2_total = r2_score(y, y_pred)

# Exibir os coeficientes do modelo
print("Coeficientes:", modelo.coef_)
print("Intercepto:", modelo.intercept_)
print("Erro Quadrático Médio (MSE):", mse)
print("Coeficiente de Determinação (R²):", r2_total)

# Calcular a matriz de correlação de Pearson
correlation_matrix = dataset[colunas].corr(method='pearson')

# ANOVA
# Ajustar o modelo de regressão linear
modelo.fit(x, y)

# Realizar ANOVA
F, p = f_regression(x, y)

# Calcular os graus de liberdade
gl_regressao = x.shape[1]
gl_residuals = x.shape[0] - x.shape[1] - 1
gl_total = x.shape[0] - 1

# Calcular a soma dos quadrados
SQT = np.sum((y - np.mean(y))**2)

SQR = np.sum((y - y_pred)**2)
SQD = SQT - SQR

# Calcular as médias quadráticas
MQR = SQD / gl_regressao
MQD = SQR / gl_residuals

# Calcular a razão F
f_value = MQR / MQD

# Criar um DataFrame com os resultados da ANOVA
anova_table = pd.DataFrame({
    'Graus de Liberdade': [gl_regressao, gl_residuals, gl_total],
    'Soma dos Quadrados': [SQD, SQR, SQT],
    'Médias Quadráticas': [MQR, MQD, np.nan],
    'Razão F': [f_value, np.nan, np.nan],
    'Teste F (p-valor)': [p[0], np.nan, np.nan]
}, index=['Modelo', 'Resíduos', 'Total'])

############# Cor
# Definir a variável dependente y (primeira coluna)
y = dataset['P.E.(Y)']
# Definir as variáveis independentes x (todas as demais colunas)
x = dataset[['Cor(X2)']]

# Criar o modelo_cor de regressão linear
modelo_cor = LinearRegression()

# Ajustar o modelo_cor aos dados
modelo_cor.fit(x, y)

# Coeficientes
intercepto = modelo_cor.intercept_
coeficiente_angular = modelo_cor.coef_[0]

# Calcular o R^2
r2_Cor = modelo_cor.score(x, y)

############# Cor + NS
# Definir a variável dependente y (primeira coluna)
y = dataset['P.E.(Y)']
# Definir as variáveis independentes x (todas as demais colunas)
x = dataset[['Cor(X2)','NS(X4)']]

# Criar o modelo_cor_NS de regressão linear
modelo_cor_NS = LinearRegression()

# Ajustar o modelo_cor_NS aos dados
modelo_cor_NS.fit(x, y)

# Coeficientes
intercepto = modelo_cor_NS.intercept_
coeficiente_angular = modelo_cor_NS.coef_[0]

# Calcular o R^2
r2_Cor_NS = modelo_cor_NS.score(x, y)

############# Cor + NS + Quartzo
# Definir a variável dependente y (primeira coluna)
y = dataset['P.E.(Y)']
# Definir as variáveis independentes x (todas as demais colunas)
x = dataset[['Cor(X2)','NS(X4)','Quartzo(X1)']]

# Criar o modelo_cor_NS_Quartzo de regressão linear
modelo_cor_NS_Quartzo = LinearRegression()

# Ajustar o modelo_cor_NS_Quartzo aos dados
modelo_cor_NS_Quartzo.fit(x, y)

# Coeficientes
intercepto = modelo_cor_NS_Quartzo.intercept_
coeficiente_angular = modelo_cor_NS_Quartzo.coef_[0]

# Calcular o R^2
r2_Cor_NS_Quartzo = modelo_cor_NS_Quartzo.score(x, y)

############# Cor + NS + Quartzo + Feldspato
# Definir a variável dependente y (primeira coluna)
y = dataset['P.E.(Y)']
# Definir as variáveis independentes x (todas as demais colunas)
x = dataset[['Cor(X2)','NS(X4)','Quartzo(X1)','Feldspato(X3)']]

# Criar o modelo_cor_NS_Quartzo_Feldspato de regressão linear
modelo_cor_NS_Quartzo_Feldspato = LinearRegression()

# Ajustar o modelo_cor_NS_Quartzo_Feldspato aos dados
modelo_cor_NS_Quartzo_Feldspato.fit(x, y)

# Coeficientes
intercepto = modelo_cor_NS_Quartzo_Feldspato.intercept_
coeficiente_angular = modelo_cor_NS_Quartzo_Feldspato.coef_[0]

# Calcular o R^2
r2_Cor_NS_Quartzo_Feldspato = modelo_cor_NS_Quartzo_Feldspato.score(x, y)

contrib_Cor = r2_Cor
contrib_NS = r2_Cor_NS - contrib_Cor 
contrib_Quartzo = r2_Cor_NS_Quartzo - contrib_NS - contrib_Cor
contrib_Feldspato = r2_Cor_NS_Quartzo_Feldspato - contrib_Quartzo - contrib_NS - contrib_Cor
contrib_EW = r2_total - contrib_Feldspato - contrib_Quartzo - contrib_NS - contrib_Cor

#### Plotagem mapas
# Criar uma grade de coordenadas x e y
x = dataset['EW(X5)']
y = dataset['NS(X4)']
z = dataset['P.E.(Y)']

# Plotar peso específico
# Criar uma grade de pontos para interpolação
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolar os valores z na grade de pontos
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Criar o mapa de contorno colorido
plt.contourf(xi, yi, zi, levels=14, cmap='coolwarm')

# Adicionar uma barra de cores
plt.colorbar()

# Adicionar rótulos e título
plt.xlabel('EW(X5)')
plt.ylabel('NS(X4)')
plt.title('Peso específico (g/cm³)')

# Mostrar o gráfico
plt.show()

# Plotar máficos
# Criar uma grade de coordenadas x e y
x = dataset['EW(X5)']
y = dataset['NS(X4)']
z = dataset['Cor(X2)']

# Criar uma grade de pontos para interpolação
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolar os valores z na grade de pontos
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Criar o mapa de contorno colorido
plt.contourf(xi, yi, zi, levels=14, cmap='coolwarm')

# Adicionar uma barra de cores
plt.colorbar()

# Adicionar rótulos e título
plt.xlabel('EW(X5)')
plt.ylabel('NS(X4)')
plt.title('Cor - Máficos')

# Mostrar o gráfico
plt.show()

# Plotar Quartzo
# Criar uma grade de coordenadas x e y
x = dataset['EW(X5)']
y = dataset['NS(X4)']
z = dataset['Quartzo(X1)']

# Criar uma grade de pontos para interpolação
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolar os valores z na grade de pontos
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Criar o mapa de contorno colorido
plt.contourf(xi, yi, zi, levels=14, cmap='coolwarm')

# Adicionar uma barra de cores
plt.colorbar()

# Adicionar rótulos e título
plt.xlabel('EW(X5)')
plt.ylabel('NS(X4)')
plt.title('Quartzo')

# Mostrar o gráfico
plt.show()

# Plotar Feldspato
# Criar uma grade de coordenadas x e y
x = dataset['EW(X5)']
y = dataset['NS(X4)']
z = dataset['Feldspato(X3)']

# Criar uma grade de pontos para interpolação
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolar os valores z na grade de pontos
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Criar o mapa de contorno colorido
plt.contourf(xi, yi, zi, levels=14, cmap='coolwarm')

# Adicionar uma barra de cores
plt.colorbar()

# Adicionar rótulos e título
plt.xlabel('EW(X5)')
plt.ylabel('NS(X4)')
plt.title('Feldspato')

# Mostrar o gráfico
plt.show()