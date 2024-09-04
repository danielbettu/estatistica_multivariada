#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:02:18 2024

@author: bettu
"""
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# URL do arquivo CSV
url = "https://raw.githubusercontent.com/danielbettu/estatistica_multivariada/main/matriz_2_1_Landim.csv"

# Carregar o CSV em um DataFrame
dataset = pd.read_csv(url)

X = dataset[['EW(X5)','NS(X4)']].values
Y = dataset['P.E.(Y)'].values

def plot_trend_surface(X, Y, degree, ax):
    # Cria características polinomiais de grau especificado
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    # Ajusta um modelo de regressão linear aos dados transformados
    model = LinearRegression().fit(X_poly, Y)
    Y_pred = model.predict(X_poly)
    
    # Criar uma grade para a superfície de tendência
    x_surf = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_surf = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    
    # Prediz os valores Z para a grade criada
    z_surf = model.predict(poly.transform(np.c_[x_surf.ravel(), y_surf.ravel()])).reshape(x_surf.shape)
    
    # Plotar o mapa de contorno da superfície de tendência
    contour = ax.contourf(x_surf, y_surf, z_surf, cmap='terrain', alpha=0.7)
    fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5)
    
    # Plotar os pontos de dados originais com uma legenda de cores
    scatter = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='terrain', edgecolors='k', label='Dados Originais')
    ax.set_title(f'Mapa de Contorno - Ordem {degree}')
    ax.set_xlabel('EW(X5)')
    ax.set_ylabel('NS(X4)')
    ax.legend()

# Cria uma figura com 3 subplots lado a lado
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, degree in enumerate([1, 2, 3]):
    plot_trend_surface(X, Y, degree, axs[i])

plt.tight_layout()
plt.show()

def calcular_resultados_estatisticos(X, Y, degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, Y)
    Y_pred = model.predict(X_poly)
    
    mse = mean_squared_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)
    
    return mse, r2

# Abrir arquivo para gravação dos resultados
with open('resultados_mapa_tendencia.txt', 'w') as file:
    file.write('Resultados Estatísticos das Superfícies Geradas\n')
    file.write('=============================================\n')
    
    for degree in [1, 2, 3]:
        mse, r2 = calcular_resultados_estatisticos(X, Y, degree)
        file.write(f'Ordem {degree}:\n')
        file.write(f'Mean Squared Error (MSE): {mse:.4f}\n')
        file.write(f'R-squared (R2): {r2:.4f}\n')
        file.write('---------------------------------------------\n')

# Nova figura tridimensional
fig = plt.figure(figsize=(18, 6))

for i, degree in enumerate([1, 2, 3]):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, Y)
    
    # Criar uma grade para a superfície
    x_surf = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_surf = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = model.predict(poly.transform(np.c_[x_surf.ravel(), y_surf.ravel()])).reshape(x_surf.shape)
    
    # Plotar a superfície
    ax.plot_surface(x_surf, y_surf, z_surf, cmap='terrain', alpha=0.7)
    
    # Plotar os pontos originais
    ax.scatter(X[:, 0], X[:, 1], Y, c=Y, cmap='terrain', edgecolors='k')
    
    ax.set_title(f'Superfície de Tendência - Ordem {degree}')
    ax.set_xlabel('EW(X5)')
    ax.set_ylabel('NS(X4)')
    ax.set_zlabel('P.E.(Y)')

plt.tight_layout()
plt.show()

