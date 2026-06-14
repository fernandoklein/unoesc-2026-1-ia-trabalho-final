"""
Etapa B - Servidor backend (FastAPI).

Carrega o modelo exportado pela Etapa A e expõe:
  - GET  /health    -> status e nome do modelo carregado
  - POST /predict   -> apenas a predição do modelo
  - POST /analisar  -> predição + explicação do agente (Gemini)

Execução (a partir da raiz do projeto):
    uvicorn backend.main:app --reload --port 8000
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Garante que predicao_utils (na raiz) seja importável para o joblib recarregar
# o RegressaoLinearClassificador, caso ele seja o modelo vencedor.
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import predicao_utils  # noqa: F401, E402 - necessário para o unpickle do modelo
from backend.agente import explicar_predicao  # noqa: E402
from backend.schemas import (  # noqa: E402
    EntradaVinho,
    RespostaAnalise,
    RespostaPredicao,
)

ARTEFATOS = RAIZ / "ml" / "artefatos"

app = FastAPI(title="Agente Preditivo de Qualidade de Vinhos", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _carregar_artefatos():
    modelo_path = ARTEFATOS / "modelo.pkl"
    scaler_path = ARTEFATOS / "scaler.pkl"
    meta_path = ARTEFATOS / "metadata.json"
    if not modelo_path.exists():
        raise RuntimeError(
            "Modelo não encontrado. Rode 'python ml/treino.py' antes de subir a API."
        )
    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, scaler, metadata


MODELO, SCALER, METADATA = _carregar_artefatos()
FEATURES = METADATA["features"]


def _predizer(entrada: EntradaVinho):
    # Reordena os valores na ordem exata em que o modelo foi treinado.
    dados = entrada.model_dump(by_alias=True)
    vetor = np.array([[dados[col] for col in FEATURES]])
    vetor = SCALER.transform(vetor)

    classe = int(MODELO.predict(vetor)[0])
    proba = float(MODELO.predict_proba(vetor)[0][1])
    rotulo = "Bom" if classe == 1 else "Ruim"
    return classe, rotulo, proba, dados


@app.get("/health")
def health():
    return {"status": "ok", "modelo": METADATA["melhor_modelo"]}


@app.post("/predict", response_model=RespostaPredicao)
def predict(entrada: EntradaVinho):
    classe, rotulo, proba, _ = _predizer(entrada)
    return RespostaPredicao(
        classe=classe,
        rotulo=rotulo,
        probabilidade=round(proba, 4),
        modelo=METADATA["melhor_modelo"],
    )


@app.post("/analisar", response_model=RespostaAnalise)
def analisar(entrada: EntradaVinho):
    try:
        classe, rotulo, proba, dados = _predizer(entrada)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro na predição: {exc}")

    explicacao = explicar_predicao(classe, proba, dados)
    return RespostaAnalise(
        classe=classe,
        rotulo=rotulo,
        probabilidade=round(proba, 4),
        modelo=METADATA["melhor_modelo"],
        explicacao=explicacao,
    )
