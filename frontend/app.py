"""
Etapa C - Interface web (Streamlit).

Duas abas:
  - Prever Qualidade: formulário com as 11 variáveis -> /analisar -> resultado
    bruto do modelo + explicação do agente (Gemini).
  - Análise dos Dados & Modelos: gráficos exploratórios (Seaborn) gerados na
    Etapa A + tabela comparativa dos quatro algoritmos.

Execução (a partir da raiz do projeto):
    streamlit run frontend/app.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "ml" / "artefatos" / "figuras"
ARTEFATOS = RAIZ / "ml" / "artefatos"

# Valores-padrão = aproximadamente a mediana do dataset (vinho "típico").
CAMPOS = [
    ("fixed acidity", "Acidez fixa", 7.9, 0.0, 20.0, 0.1),
    ("volatile acidity", "Acidez volátil", 0.52, 0.0, 2.0, 0.01),
    ("citric acid", "Ácido cítrico", 0.26, 0.0, 1.5, 0.01),
    ("residual sugar", "Açúcar residual", 2.2, 0.0, 20.0, 0.1),
    ("chlorides", "Cloretos", 0.079, 0.0, 1.0, 0.001),
    ("free sulfur dioxide", "Dióxido de enxofre livre", 14.0, 0.0, 100.0, 1.0),
    ("total sulfur dioxide", "Dióxido de enxofre total", 38.0, 0.0, 300.0, 1.0),
    ("density", "Densidade", 0.9968, 0.9, 1.1, 0.0001),
    ("pH", "pH", 3.31, 2.0, 5.0, 0.01),
    ("sulphates", "Sulfatos", 0.62, 0.0, 2.0, 0.01),
    ("alcohol", "Álcool (%)", 10.2, 5.0, 20.0, 0.1),
]

st.set_page_config(
    page_title="Qualidade de Vinhos",
    page_icon="🍷",
    layout="wide",
)

# Estilo enxuto e neutro: usa as variáveis de tema do Streamlit (--text-color,
# --secondary-background-color, etc.) para funcionar igual em light e dark.
st.markdown(
    """
    <style>
      :root { --vinho:#7b1e3b; }

      /* Esconde o botão Deploy e deixa o header nativo transparente (mantém o
         menu de 3 pontinhos flutuante para trocar tema / configs). */
      [data-testid="stAppDeployButton"] { display:none; }
      header[data-testid="stHeader"] { background:transparent; box-shadow:none; }
      .block-container { padding-top:3rem; }

      /* Topbar */
      .navbar {
        display:flex; align-items:center; justify-content:space-between;
        background:linear-gradient(90deg,#511025,#7b1e3b 62%,#8f2a4d);
        color:#fff; padding:15px 24px; border-radius:12px;
        margin-bottom:24px; box-shadow:0 3px 14px rgba(81,16,37,.22);
      }
      .navbar .brand { display:flex; align-items:center; gap:13px; }
      .navbar .logo { font-size:1.7rem; line-height:1; }
      .navbar .name {
        font-family:Georgia,"Times New Roman",serif; font-weight:600;
        font-size:1.4rem; line-height:1.1;
      }
      .navbar .name small {
        display:block; font-weight:400; font-size:.72rem; opacity:.82;
        letter-spacing:.03em; margin-top:2px;
      }
      .navbar .badge {
        background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.3);
        padding:6px 15px; border-radius:99px; font-size:.82rem; white-space:nowrap;
      }
      @media (max-width:640px){ .navbar .badge{ display:none; } }

      .secao {
        font-family:Georgia, serif; font-size:1.15rem; font-weight:600;
        color:var(--text-color); margin:4px 0 10px;
      }

      /* Resultado: cor própria + texto branco -> legível nos dois temas. */
      .resultado { border-radius:10px; padding:20px 22px; color:#fff; }
      .res-bom  { background:#2e6e34; }
      .res-ruim { background:#9a3b22; }
      .resultado .topo-res { font-size:.82rem; letter-spacing:.04em; text-transform:uppercase; opacity:.85; }
      .resultado .rotulo { font-size:2rem; font-weight:700; margin:2px 0 8px; }
      .resultado .pct { font-size:.95rem; opacity:.95; }
      .barra-out { background:rgba(255,255,255,.28); border-radius:99px; height:10px; margin-top:10px; overflow:hidden; }
      .barra-in  { background:#fff; height:100%; }

      /* Explicação: segue o tema. */
      .explica {
        background:var(--secondary-background-color);
        color:var(--text-color);
        border-left:3px solid var(--vinho); border-radius:8px;
        padding:16px 18px; margin-top:16px; line-height:1.6;
      }
      .explica .titulo { font-weight:600; margin-bottom:8px; display:block; }

      .stButton>button {
        background:var(--vinho); color:#fff; border:none; border-radius:8px;
        padding:9px 18px; font-weight:600; width:100%;
      }
      .stButton>button:hover { filter:brightness(1.1); color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def carregar_metadata():
    meta_path = ARTEFATOS / "metadata.json"
    comp_path = ARTEFATOS / "comparativo.csv"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    comp = pd.read_csv(comp_path) if comp_path.exists() else None
    return meta, comp


@st.cache_data
def carregar_presets():
    """Medianas reais de cada classe, usadas pelos botões de exemplo."""
    csv = RAIZ / "data" / "winequality-red.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["label"] = (df["quality"] >= 7).astype(int)
    cols = [c[0] for c in CAMPOS]
    return {
        "bom": df[df.label == 1][cols].median().round(3).to_dict(),
        "tipico": df[cols].median().round(3).to_dict(),
        "ruim": df[df.label == 0][cols].median().round(3).to_dict(),
    }


metadata, comparativo = carregar_metadata()
presets = carregar_presets()
melhor_modelo = metadata.get("melhor_modelo", "—")

# Inicializa o estado dos campos (necessário para os botões de exemplo poderem
# preenchê-los antes de os number_input serem criados).
for chave, _rot, padrao, *_ in CAMPOS:
    st.session_state.setdefault(f"in_{chave}", float(padrao))


def aplicar_preset(perfil):
    if presets and perfil in presets:
        for chave, *_ in CAMPOS:
            st.session_state[f"in_{chave}"] = float(presets[perfil][chave])

st.markdown(
    f"""
    <div class="navbar">
      <div class="brand">
        <span class="logo">🍷</span>
        <span class="name">Qualidade de Vinhos
          <small>Análise físico-química com aprendizado de máquina</small>
        </span>
      </div>
      <span class="badge">Modelo ativo · {melhor_modelo}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

aba_prever, aba_analise = st.tabs(["Prever qualidade", "Dados e modelos"])

# ================================================================ ABA 1
with aba_prever:
    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown('<div class="secao">Propriedades do vinho</div>', unsafe_allow_html=True)

        if presets:
            st.caption("Carregar um exemplo (medianas reais de cada classe):")
            b1, b2, b3 = st.columns(3)
            b1.button("Vinho bom", width="stretch", on_click=aplicar_preset, args=("bom",))
            b2.button("Mediano", width="stretch", on_click=aplicar_preset, args=("tipico",))
            b3.button("Vinho ruim", width="stretch", on_click=aplicar_preset, args=("ruim",))

        with st.form("formulario"):
            sub = st.columns(2)
            for i, (chave, rotulo, _padrao, minimo, maximo, passo) in enumerate(CAMPOS):
                with sub[i % 2]:
                    st.number_input(
                        rotulo,
                        min_value=float(minimo),
                        max_value=float(maximo),
                        step=float(passo),
                        format="%.4f",
                        key=f"in_{chave}",
                    )
            enviar = st.form_submit_button("Analisar vinho")

        valores = {chave: st.session_state[f"in_{chave}"] for chave, *_ in CAMPOS}

    with col_result:
        st.markdown('<div class="secao">Resultado</div>', unsafe_allow_html=True)
        if not enviar:
            st.info("Preencha os campos e clique em **Analisar vinho**.")
        else:
            with st.spinner("Consultando o modelo e o agente..."):
                try:
                    resp = requests.post(f"{BACKEND_URL}/analisar", json=valores, timeout=60)
                    resp.raise_for_status()
                    dados = resp.json()
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Não foi possível conectar ao backend em {BACKEND_URL}. "
                        "Confira se a API está rodando (uvicorn backend.main:app)."
                    )
                    st.stop()
                except requests.exceptions.HTTPError as exc:
                    st.error(f"Erro do servidor: {exc.response.text}")
                    st.stop()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erro inesperado: {exc}")
                    st.stop()

            rotulo = dados["rotulo"]
            prob = dados["probabilidade"]
            classe = dados["classe"]
            css = "res-bom" if classe == 1 else "res-ruim"

            st.markdown(
                f"""
                <div class="resultado {css}">
                  <div class="topo-res">Predição · {dados['modelo']}</div>
                  <div class="rotulo">{rotulo}</div>
                  <div class="pct">Probabilidade de ser bom: <b>{prob:.0%}</b></div>
                  <div class="barra-out"><div class="barra-in" style="width:{prob*100:.0f}%"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            explicacao_html = dados["explicacao"].replace(chr(10), "<br>")
            st.markdown(
                f"""
                <div class="explica">
                  <span class="titulo">Interpretação do agente</span>{explicacao_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

# ================================================================ ABA 2
with aba_analise:
    st.markdown('<div class="secao">Comparação dos modelos</div>', unsafe_allow_html=True)
    if comparativo is not None:
        df = comparativo.copy()
        col_metricas = [c for c in df.columns if c != "Modelo"]
        estilo = (
            df.style.format({c: "{:.3f}" for c in col_metricas})
            .highlight_max(subset=col_metricas, color="#2e6e34")
            .set_properties(subset=col_metricas, **{"text-align": "center"})
        )
        st.dataframe(estilo, width="stretch", hide_index=True)
        st.caption(
            f"Modelo exportado (melhor F1): {melhor_modelo}. "
            "Destaque = melhor valor de cada métrica."
        )
    else:
        st.warning("Rode `python ml/treino.py` para gerar a tabela comparativa.")

    st.divider()
    st.markdown('<div class="secao">Análise exploratória</div>', unsafe_allow_html=True)

    figuras = [
        ("correlacao.png", "Matriz de correlação",
         "Álcool tem correlação positiva e acidez volátil negativa com a qualidade."),
        ("distribuicao.png", "Distribuição da qualidade",
         "Classes desbalanceadas — predominam vinhos de qualidade baixa (0)."),
        ("boxplot.png", "Boxplot das variáveis",
         "Presença de outliers, principalmente em dióxido de enxofre."),
        ("comparativo.png", "Acurácia por modelo",
         "Comparação visual da acurácia dos quatro algoritmos."),
    ]
    cols = st.columns(2, gap="large")
    for i, (arquivo, titulo, legenda) in enumerate(figuras):
        caminho = FIGURAS / arquivo
        with cols[i % 2]:
            st.markdown(f"**{titulo}**")
            if caminho.exists():
                st.image(str(caminho), width="stretch")
                st.caption(legenda)
            else:
                st.warning(f"`{arquivo}` não encontrado. Rode `python ml/treino.py`.")
