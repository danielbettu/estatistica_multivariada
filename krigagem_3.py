#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:59:54 2024

@author: bettu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skgstat import Variogram, Kriging, OrdinaryKriging  # Importar diretamente a classe Kriging
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import Polynomial

# URL do arquivo CSV
url = "https://raw.githubusercontent.com/GeostatsGuy/GeoDataSets/refs/heads/master/2D_MV_200wells.csv"

# Carregar o CSV em um DataFrame
data = pd.read_csv(url)

'''
# Variáveis independentes
X_reg = data[['X', 'Y']]
# Variável dependente
y_reg = data['porosity']

# Criar o modelo de regressão linear
model = LinearRegression()
# Ajustar o modelo aos dados
model.fit(X_reg, y_reg)

# Coeficientes da regressão
print('Coeficientes a regressão linear múltipla:', model.coef_)
print('Intercepto:', model.intercept_)
'''
x = data['X'].values
y = data['Y'].values
z = data['porosity'].values

# Matriz de design com termos até a 3ª ordem
A = np.column_stack([x**3, y**3, x**2 * y, x * y**2, x**2, y**2, x * y, x, y, np.ones(x.shape)])

# Resolva o sistema para encontrar os coeficientes
coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

# Valores previstos (superfície de tendência) nas posições originais
z_tendencia = (coeffs[0] * x**3 + coeffs[1] * y**3 +
               coeffs[2] * x**2 * y + coeffs[3] * x * y**2 +
               coeffs[4] * x**2 + coeffs[5] * y**2 +
               coeffs[6] * x * y + coeffs[7] * x +
               coeffs[8] * y + coeffs[9])

residuos = z - z_tendencia

# Exploração dos dados
plt.scatter(data['X'], data['Y'], c=data['porosity'], cmap='inferno', alpha=1, edgecolor='gray', s=data['porosity'] * 150)
plt.colorbar(label='Porosidade')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Dados Originais')
plt.show()

x_coord_data = data['X']
y_coord_data = data['Y']

# # Criando uma grade de pontos regulares
# x_min, x_max = x_coord_data.min(), x_coord_data.max()
# y_min, y_max = y_coord_data.min(), y_coord_data.max()
# xx_data, yy_data = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

# Criando uma grade de pontos originais
xx_data, yy_data = data['X'], data['Y']

'''
# Prevendo os valores de porosidade para os pontos originais
Z_estimado_dados = model.predict(np.c_[xx_data.ravel(), yy_data.ravel()])
Z_estimado_dados = Z_estimado_dados.reshape(xx_data.shape)

residuos = Z_estimado_dados - y_reg
'''

#histograma dos resíduos
plt.hist(residuos, bins=15)
plt.show()

# Criando uma grade de pontos regulares
x_min, x_max = x_coord_data.min(), x_coord_data.max()
y_min, y_max = y_coord_data.min(), y_coord_data.max()
xx_previsao, yy_previsao = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

'''
# Prevendo os valores de porosidade para a malha regular a partir da sup de tendência
Z_estimado_regressao = model.predict(np.c_[xx_previsao.ravel(), yy_previsao.ravel()])
Z_estimado_regressao = Z_estimado_regressao.reshape(xx_previsao.shape)  # Ajustando o formato para 50x50
'''
x = xx_previsao
y = yy_previsao

# plotagem dispersão - resíduos
plt.title('Diagrama de dispersão dos resíduos')
plt.scatter(data['X'], data['Y'], c=residuos, cmap='inferno', alpha=1, edgecolor='gray', s=data['porosity'] * 150)
plt.colorbar(label='porosidade (valor residual)')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.show()

# superfície de tendência nos pontos para predição
z_tendencia_predicao = (coeffs[0] * x**3 + coeffs[1] * y**3 +
               coeffs[2] * x**2 * y + coeffs[3] * x * y**2 +
               coeffs[4] * x**2 + coeffs[5] * y**2 +
               coeffs[6] * x * y + coeffs[7] * x +
               coeffs[8] * y + coeffs[9])

# Plotando o mapa de contorno da tendência para a malha regular
plt.contourf(xx_previsao, yy_previsao, z_tendencia_predicao, levels=15, cmap='gray')
plt.colorbar(label='Porosidade')
plt.title('Superfície de Tendência de 1º grau')
plt.scatter(data['X'], data['Y'], c=residuos, cmap='hot', alpha=1, edgecolor='gray', s=data['porosity'] * 150)
scatter = plt.scatter(data['X'], data['Y'], c=residuos, cmap='seismic', alpha=1, edgecolor='gray', s=data['porosity'] * 150)
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Superfície de Tendência')
plt.colorbar(scatter, label='Resíduos')  # Barra de cores para os resíduos
plt.show()


# estimando a distância máxima do variograma
x_min, x_max = x_coord_data.min(), x_coord_data.max()
y_min, y_max = y_coord_data.min(), y_coord_data.max()

# # Calculando a distância máxima para o variograma
# dist_max_variograma = ((x_max - x_min)**2 + (y_max - y_min)**2)**0.5 / 3

# Create the variogram object
variograma_experimental = Variogram(data[['X', 'Y']], residuos, maxlag=0.3, n_lags=13, model='gaussian', normalize=False)

# Ajustando um modelo variográfico e plotando
variograma_experimental.fit()
variograma_experimental.plot()

# Extraindo o número de pares e os valores do variograma
bin_centers = variograma_experimental.bins  # Centros das classes (lags)
bin_counts = variograma_experimental.bin_count  # Número de pares para cada lag
variogram_values = variograma_experimental.experimental  # Valores experimentais do variograma

# Plotando o variograma
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, variogram_values, 'o-', label='Variograma Experimental dos Resíduos')
plt.xlabel('Distância (lag)')
plt.ylabel('Semivariância')
plt.title('Variograma Experimental dos resíduos com Número de Pares')
# Adicionando o número de pares para cada ponto
for x, y, count in zip(bin_centers, variogram_values, bin_counts):
    plt.text(x, y, f'{count}', ha='center', va='bottom', fontsize=9, color='blue')

plt.legend()
plt.show()

# Krigagem
ok = OrdinaryKriging(variograma_experimental, min_points=1)
# Flatten a malha para alimentar a krigagem
grid_points = np.vstack([xx_previsao.ravel(), yy_previsao.ravel()]).T
# Fazer a krigagem para estimar os valores nos pontos do grid
estimativa_residuos = ok.transform(grid_points)
# Remodelar o array resultante para que combine com o grid original
residuo_estimado_krigado = estimativa_residuos.reshape(xx_previsao.shape)
# Adiciona aos resíduos o valor da superfície de tendência
estimativa_final_krigagem = z_tendencia_predicao + residuo_estimado_krigado
'''
# Plotar a imagem
plt.imshow(estimativa_final_krigagem, extent=(xx_previsao.min(), xx_previsao.max(), yy_previsao.min(), yy_previsao.max()), origin='lower', cmap='terrain')
plt.colorbar(label='Estimativa de Krigagem')
scatter_original = plt.scatter(data['X'], data['Y'], c=data['porosity'], cmap='terrain', alpha=1, edgecolor='black', s=data['porosity'] * 150)
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Mapa de Estimativas Krigado')
plt.colorbar(scatter_original, label='Porosidade original')  # Barra de cores para os resíduos
plt.show()
'''
# Determinar o limite combinado de valores para manter a mesma escala, ignorando NaNs
vmin = min(np.nanmin(estimativa_final_krigagem), np.nanmin(data['porosity']))
vmax = max(np.nanmax(estimativa_final_krigagem), np.nanmax(data['porosity']))

# Plotar a estimativa de krigagem
plt.figure(figsize=(20, 12)) # Definir o tamanho da figura
plt.imshow(estimativa_final_krigagem, extent=(xx_previsao.min(), xx_previsao.max(), yy_previsao.min(), yy_previsao.max()), origin='lower', cmap='terrain', vmin=vmin, vmax=vmax)
plt.colorbar(label='Estimativa de Krigagem')

# Plotar os dados originais de porosidade
scatter_original = plt.scatter(data['X'], data['Y'], c=data['porosity'], cmap='terrain', vmin=vmin, vmax=vmax, alpha=1, edgecolor='black', s=data['porosity'] * 150)
plt.colorbar(scatter_original, label='Porosidade Original')

# Configurações dos eixos e título
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Mapa de Estimativas Krigado')
plt.show()

