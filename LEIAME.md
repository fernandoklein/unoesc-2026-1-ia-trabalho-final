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

Projeto desenvolvido individualmente por **Fernando Germano Klein**, em etapas,
ao longo dos dias de desenvolvimento.

| Data | Atividade |
|------|-----------|
| **08/06** | Abertura do trabalho: definição do escopo e organização do repositório (pastas `ml/`, `backend/`, `frontend/`). Reaproveitei a base *Wine Quality* (vinho tinto) usada no Trabalho I, criei o ambiente virtual e o `requirements.txt`. |
| **09/06** | Pré-processamento dos dados: leitura do CSV, verificação de valores nulos e criação da variável alvo binária (`quality >= 7`). Revisei o que fiz no Trabalho I e corrigi o ajuste do `StandardScaler` para ser feito apenas no treino, com divisão estratificada treino/teste. |
| **10/06** | Análise exploratória com Seaborn: matriz de correlação, boxplot e gráfico de distribuição. Confirmei o desbalanceamento das classes e a relação do álcool (positiva) e da acidez volátil (negativa) com a qualidade. |
| **11/06** | Modelagem: implementei os quatro algoritmos exigidos — Regressão Linear Múltipla, KNN, MLP e Naive Bayes — com ajuste de hiperparâmetros (GridSearch) onde aplicável. Precisei adaptar a Regressão Linear para classificação usando um limiar de 0,5. |
| **12/06** | Avaliação dos modelos pelas métricas de acurácia, precisão, sensibilidade e especificidade. Montei a tabela comparativa, escolhi o melhor modelo pelo F1 (KNN) e exportei `modelo.pkl`, `scaler.pkl` e o `metadata.json`. |
| **13/06** | Backend com FastAPI (rotas `/predict` e `/analisar`, validação das 11 variáveis com Pydantic) e integração com o agente inteligente (Gemini): *System Prompt* para explicar a predição sem alucinar, tratamento de erros/timeout e modo de contingência. |
| **14/06** | Frontend com Streamlit (formulário, exemplos por classe e visualização dos gráficos e da comparação dos modelos), melhorias na interface, documentação (este README), testes finais de ponta a ponta e publicação do repositório no GitHub. |
