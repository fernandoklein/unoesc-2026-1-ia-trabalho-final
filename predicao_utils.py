"""
Utilitários compartilhados entre o treino (ml/treino.py) e o backend.

A classe RegressaoLinearClassificador precisa estar em um módulo importável por
ambos: o joblib/pickle grava o caminho do módulo da classe, então o backend só
consegue recarregar o modelo se conseguir importar a classe pelo mesmo caminho
(predicao_utils.RegressaoLinearClassificador).
"""

import numpy as np


class RegressaoLinearClassificador:
    """Adapta um LinearRegression para classificação binária por limiar (0.5),
    expondo predict/predict_proba para uniformizar o uso no backend."""

    def __init__(self, modelo_linear, limiar=0.5):
        self.modelo = modelo_linear
        self.limiar = limiar

    def predict(self, X):
        return (self.modelo.predict(X) >= self.limiar).astype(int)

    def predict_proba(self, X):
        # Aproxima uma "probabilidade" recortando a saída linear em [0, 1].
        bruto = np.clip(self.modelo.predict(X), 0.0, 1.0)
        return np.column_stack([1 - bruto, bruto])
