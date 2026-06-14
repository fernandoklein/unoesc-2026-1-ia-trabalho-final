"""
Etapa A - Preparação, modelagem e comparação de modelos.

Treina e compara quatro algoritmos exigidos pelo enunciado do Trabalho II
(Regressão Linear Múltipla, KNN, MLP e Naive Bayes) na base Wine Quality
(vinho tinto), gera os gráficos exploratórios com Seaborn e exporta o modelo
de melhor desempenho para uso pelo backend.

Uso:
    python ml/treino.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Caminhos relativos à raiz do projeto (pasta acima de ml/)
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from predicao_utils import RegressaoLinearClassificador  # noqa: E402

CSV_PATH = RAIZ / "data" / "winequality-red.csv"
ARTEFATOS = RAIZ / "ml" / "artefatos"
FIGURAS = ARTEFATOS / "figuras"

LIMIAR_QUALIDADE = 7  # quality >= 7 => vinho bom (classe 1)
RANDOM_STATE = 42

# Ordem canônica das 11 variáveis físico-químicas (entrada do modelo).
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def especificidade(y_true, y_pred):
    """Especificidade = verdadeiros negativos / (VN + falsos positivos)."""
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def carregar_dados():
    df = pd.read_csv(CSV_PATH, sep=",")
    print("\n=== Head ===")
    print(df.head())
    print("\n=== Info ===")
    df.info()
    print("\n=== Valores nulos por coluna ===")
    print(df.isnull().sum())

    # Variável alvo binária.
    df["quality_label"] = (df["quality"] >= LIMIAR_QUALIDADE).astype(int)
    return df


def gerar_graficos_eda(df):
    """Gráficos exploratórios com Seaborn (correlação, boxplot, distribuição)."""
    FIGURAS.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.savefig(FIGURAS / "correlacao.png", dpi=120)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[FEATURES])
    plt.xticks(rotation=45, ha="right")
    plt.title("Boxplot das Variáveis")
    plt.tight_layout()
    plt.savefig(FIGURAS / "boxplot.png", dpi=120)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(x="quality_label", data=df)
    plt.title("Distribuição da Qualidade (0 = ruim, 1 = bom)")
    plt.tight_layout()
    plt.savefig(FIGURAS / "distribuicao.png", dpi=120)
    plt.close()

    print(f"\nGráficos exploratórios salvos em {FIGURAS}")


def preparar_dados(df):
    """Split estratificado ANTES de escalar (StandardScaler ajustado só no treino,
    evitando vazamento de dados de teste para o treino)."""
    X = df[FEATURES].values
    y = df["quality_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def treinar_modelos(X_train, y_train):
    """Treina os quatro algoritmos exigidos, com ajuste de parâmetros via GridSearch
    onde se aplica. Retorna dict {nome: estimador_treinado}."""
    modelos = {}

    # 1) Regressão Linear Múltipla — algoritmo de regressão; a saída contínua é
    #    limiarizada em 0.5 para virar classificação binária (ver classe wrapper).
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    modelos["Regressão Linear Múltipla"] = RegressaoLinearClassificador(reg)

    # 2) KNN
    grid_knn = GridSearchCV(
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
        cv=5,
        scoring="f1",
    )
    grid_knn.fit(X_train, y_train)
    modelos["KNN"] = grid_knn.best_estimator_
    print(f"\nKNN melhores params: {grid_knn.best_params_}")

    # 3) MLP (rede neural)
    grid_mlp = GridSearchCV(
        MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
        {
            "hidden_layer_sizes": [(50,), (100,), (50, 25)],
            "alpha": [0.0001, 0.001, 0.01],
        },
        cv=5,
        scoring="f1",
    )
    grid_mlp.fit(X_train, y_train)
    modelos["MLP"] = grid_mlp.best_estimator_
    print(f"MLP melhores params: {grid_mlp.best_params_}")

    # 4) Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    modelos["Naive Bayes"] = nb

    return modelos


def avaliar(modelos, X_test, y_test):
    resultados = []
    print("\n===== RESULTADOS =====")
    for nome, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        esp = especificidade(y_test, y_pred)

        print(f"\nModelo: {nome}")
        print(f"  Acurácia:       {acc:.4f}")
        print(f"  Precisão:       {prec:.4f}")
        print(f"  Sensibilidade:  {rec:.4f}")
        print(f"  Especificidade: {esp:.4f}")
        print(f"  Matriz de confusão:\n{confusion_matrix(y_test, y_pred)}")

        resultados.append(
            {
                "Modelo": nome,
                "Acurácia": acc,
                "Precisão": prec,
                "Sensibilidade": rec,
                "Especificidade": esp,
            }
        )

    return pd.DataFrame(resultados)


def grafico_comparativo(df_resultados):
    plt.figure(figsize=(9, 5))
    sns.barplot(x="Modelo", y="Acurácia", data=df_resultados)
    plt.title("Comparação de Acurácia dos Modelos")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURAS / "comparativo.png", dpi=120)
    plt.close()


def exportar_melhor(df_resultados, modelos, scaler):
    """Escolhe o melhor modelo (critério: F1 implícito via média de precisão e
    sensibilidade para lidar com o desbalanceamento) e exporta os artefatos."""
    df_resultados = df_resultados.copy()
    df_resultados["F1"] = (
        2
        * df_resultados["Precisão"]
        * df_resultados["Sensibilidade"]
        / (df_resultados["Precisão"] + df_resultados["Sensibilidade"]).replace(0, np.nan)
    ).fillna(0)

    melhor_nome = df_resultados.sort_values("F1", ascending=False).iloc[0]["Modelo"]
    melhor_modelo = modelos[melhor_nome]
    print(f"\n>>> Melhor modelo (por F1): {melhor_nome}")

    ARTEFATOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(melhor_modelo, ARTEFATOS / "modelo.pkl")
    joblib.dump(scaler, ARTEFATOS / "scaler.pkl")

    metadata = {
        "melhor_modelo": melhor_nome,
        "features": FEATURES,
        "limiar_qualidade": LIMIAR_QUALIDADE,
        "metricas": df_resultados.round(4).to_dict(orient="records"),
    }
    with open(ARTEFATOS / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    df_resultados.round(4).to_csv(ARTEFATOS / "comparativo.csv", index=False)
    print(f"Artefatos exportados em {ARTEFATOS}")


def main():
    df = carregar_dados()
    gerar_graficos_eda(df)
    X_train, X_test, y_train, y_test, scaler = preparar_dados(df)
    modelos = treinar_modelos(X_train, y_train)
    df_resultados = avaliar(modelos, X_test, y_test)

    print("\n=== Tabela Comparativa ===")
    print(df_resultados.round(4).to_string(index=False))

    grafico_comparativo(df_resultados)
    exportar_melhor(df_resultados, modelos, scaler)


if __name__ == "__main__":
    main()
