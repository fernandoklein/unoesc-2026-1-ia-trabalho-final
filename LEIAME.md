# 🍷 Agente Preditivo de Qualidade de Vinhos

Projeto da disciplina **Inteligência Artificial e Sistemas Inteligentes** (UNOESC).
Unifica exploração de dados, machine learning e IA generativa: treina modelos de
classificação na base **Wine Quality (vinho tinto)**, exporta o de melhor
desempenho e o disponibiliza por uma API com um **agente inteligente (Gemini)**
que explica a predição em linguagem natural, consumido por uma interface web.

## Arquitetura

```
Usuário (Streamlit) → POST /analisar (FastAPI)
        → modelo.pkl + scaler.pkl  → predição (Bom/Ruim + probabilidade)
        → Gemini (System Prompt)   → explicação em linguagem natural
   → interface mostra resultado bruto + explicação do agente
```

| Camada | Tecnologia | Pasta |
|---|---|---|
| Modelagem (ML) | scikit-learn, seaborn | `ml/` |
| Backend / Agente | FastAPI, google-generativeai | `backend/` |
| Frontend | Streamlit | `frontend/` |

## Modelos comparados (Etapa A)

Conforme o enunciado, foram aplicados e comparados quatro algoritmos:

- **Regressão Linear Múltipla** (saída contínua limiarizada em 0,5 para classificar)
- **KNN** (com ajuste de `n_neighbors`/`weights` via GridSearch)
- **MLP** — rede neural (ajuste de `hidden_layer_sizes`/`alpha`)
- **Naive Bayes** (GaussianNB)

Métricas: **acurácia, precisão, sensibilidade (recall) e especificidade**.
O melhor modelo (por F1, dado o desbalanceamento das classes) é exportado para
`ml/artefatos/`.

## Como rodar

### 1. Pré-requisitos
- Python 3.10+
- Chave da API do Gemini (gratuita em https://aistudio.google.com/app/apikey)

### 2. Instalação
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar a chave
```bash
cp .env.example .env       # no Windows: copy .env.example .env
# edite o .env e preencha GEMINI_API_KEY
```
> Sem a chave o app continua funcionando: a predição do modelo é normal e o
> agente entra em modo de explicação automática (fallback).

### 4. Treinar e exportar o modelo (Etapa A)
```bash
python ml/treino.py
```
Gera `ml/artefatos/modelo.pkl`, `scaler.pkl`, `metadata.json`, `comparativo.csv`
e os gráficos em `ml/artefatos/figuras/`.

### 5. Subir o backend (Etapa B)
```bash
uvicorn backend.main:app --reload --port 8000
```
Documentação interativa em http://localhost:8000/docs

### 6. Subir o frontend (Etapa C)
Em outro terminal (com o venv ativado):
```bash
streamlit run frontend/app.py
```
Acesse http://localhost:8501

## Estrutura do projeto
```
agente-preditivo-vinhos/
├── data/winequality-red.csv     # base do Kaggle (vinho tinto)
├── ml/treino.py                 # EDA + 4 algoritmos + exporta o melhor
├── ml/artefatos/                # modelo.pkl, scaler.pkl, metadata.json, figuras/
├── backend/main.py              # FastAPI (/health, /predict, /analisar)
├── backend/agente.py            # integração Gemini + System Prompt
├── backend/schemas.py           # validação das 11 variáveis
├── frontend/app.py              # interface Streamlit
├── predicao_utils.py            # classe compartilhada (wrapper da Reg. Linear)
└── requirements.txt
```

## Endpoints da API
| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | status + modelo carregado |
| POST | `/predict` | só a predição do modelo |
| POST | `/analisar` | predição + explicação do agente |

---

## Diário de Bordo de Contribuições

> Cada integrante descreve o que fez durante os 15 dias de desenvolvimento.

### Fernando Germano Klein
- **Dia 1–3 —** _(descrever)_
- **Dia 4–7 —** _(descrever)_
- **Dia 8–11 —** _(descrever)_
- **Dia 12–15 —** _(descrever)_

<!-- Adicione um bloco por integrante da equipe. -->
