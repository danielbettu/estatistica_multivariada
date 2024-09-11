#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 22:19:22 2024

@author: bettu
"""

import numpy as np
import pandas as pd
from scipy.stats import f

# Suponha que dataset1 e dataset2 sejam DataFrames pandas
dataset1 = pd.DataFrame({
    'var1': [2.1, 2.3, 2.2, 2.4],
    'var2': [3.5, 3.6, 3.7, 3.4],
    'var3': [4.2, 4.1, 4.3, 4.0]
})

dataset2 = pd.DataFrame({
    'var1': [3.1, 3.3, 3.2, 3.4],
    'var2': [4.5, 4.6, 4.7, 4.4],
    'var3': [5.2, 5.1, 5.3, 5.0]
})

# Converter DataFrames para matrizes NumPy
X1 = dataset1.to_numpy()
X2 = dataset2.to_numpy()

# Calcular as médias dos grupos
mean1 = np.mean(X1, axis=0)
mean2 = np.mean(X2, axis=0)

# Número de observações em cada grupo
n1, n2 = X1.shape[0], X2.shape[0]

# Calcular as matrizes de covariância
cov1 = np.cov(X1, rowvar=False, bias=True)
cov2 = np.cov(X2, rowvar=False, bias=True)

# Estimativa da matriz de covariância combinada
pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

# Diferença das médias
mean_diff = mean1 - mean2

# Cálculo do teste Hotelling's T²
T2 = n1 * n2 / (n1 + n2) * np.dot(np.dot(mean_diff.T, np.linalg.inv(pooled_cov)), mean_diff)

# Graus de liberdade e valor crítico do teste
p = X1.shape[1]  # número de variáveis
df1 = p
df2 = n1 + n2 - p - 1

# Valor crítico da distribuição F
critical_value = f.ppf(0.95, df1, df2)

# Preparar o texto para o arquivo
resultados = f"""
Resultados do Teste de Hotelling's T²

Matrizes de Dados:
    
- Dataset 1:
{dataset1}

- Dataset 2:
{dataset2}

Médias dos Grupos:
    
- Média do Dataset 1: {mean1}

- Média do Dataset 2: {mean2}

Matrizes de Covariância:
    
- Covariância do Dataset 1:
{cov1}

- Covariância do Dataset 2:
{cov2}

Matriz de Covariância Combinada:
{pooled_cov}

Diferença das Médias:
{mean_diff}

Resultado do Teste Hotelling's T²:
T² = {T2}

Valor Crítico (p = 0.05, distribuição F):
{critical_value}
"""

# Salvar os resultados em um arquivo de texto
with open('resultados_teste_significancia_multivariado.txt', 'w') as f:
    f.write(resultados)

print("Os resultados foram salvos em 'resultados_teste_significancia_multivariado.txt'.")

# Imprimir os resultados na tela do Spyder
print(resultados)
if T2 > critical_value:
    print("Rejeita-se a hipótese nula: as médias dos dois grupos são diferentes.")
else:
    print("Não se rejeita a hipótese nula: não há evidências suficientes para afirmar que as médias dos dois grupos são diferentes.")
