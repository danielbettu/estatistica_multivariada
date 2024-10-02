#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:09:32 2024

@author: bettu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

def perform_factor_analysis(n_factors):
    # 1. Carregar Dataset
    diabetes = load_diabetes()
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    # 2. Estandarizar Dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 3. Calcular Matriz de Correlação
    corr_matrix = np.corrcoef(data_scaled, rowvar=False)

    # 4. Aplicar Análise Fatorial
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(data_scaled)

    # 5. Cargas Fatoriais
    loadings = fa.loadings_

    # 6. Variância Explicada
    variance = fa.get_factor_variance()

    # 7. Gerar Gráficos
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title('Matriz de Correlação')
    plt.show()

    # 8. Salvar Relatório
    with open("relatorio_analise_fatorial.txt", "w") as file:
        file.write("Cargas Fatoriais:\n")
        file.write(pd.DataFrame(loadings, columns=[f'Fator {i+1}' for i in range(n_factors)]).to_string())
        file.write("\n\nVariância Explicada:\n")
        file.write(pd.DataFrame(variance, index=['Variância', 'Proporção', 'Proporção Acumulada'], columns=[f'Fator {i+1}' for i in range(n_factors)]).to_string())

    # Armazenar variáveis em um dicionário
    results = {
        'data_scaled': data_scaled,
        'corr_matrix': corr_matrix,
        'loadings': loadings,
        'variance': variance,
        'diabetes': diabetes
    }

    print("Relatório salvo como relatorio_analise_fatorial.txt")

    return results

# Defina o número de fatores que você quer analisar
n_factors = int(input("Digite o número de fatores: "))
results = perform_factor_analysis(n_factors)

# Agora você pode acessar os resultados através do dicionário 'results'
print(results)
