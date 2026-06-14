"""
Gera o Relatório Técnico (PDF) do Trabalho II, seguindo a estrutura do Trabalho I
e acrescentando a seção de arquitetura do agente inteligente.

Uso:
    python relatorio/gerar_relatorio.py
Saída:
    relatorio/Relatorio_Tecnico.pdf
"""

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "ml" / "artefatos" / "figuras"
ARTEFATOS = RAIZ / "ml" / "artefatos"
SAIDA = RAIZ / "relatorio" / "Relatorio_Tecnico.pdf"

LARGURA_UTIL = A4[0] - 4 * cm  # margens de 2cm

# ---------------------------------------------------------------- estilos
styles = getSampleStyleSheet()
body = ParagraphStyle(
    "corpo", parent=styles["Normal"], fontName="Times-Roman", fontSize=11,
    leading=16, alignment=TA_JUSTIFY, spaceAfter=8,
)
h1 = ParagraphStyle(
    "h1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=14,
    spaceBefore=14, spaceAfter=8,
)
h2 = ParagraphStyle(
    "h2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=12,
    spaceBefore=10, spaceAfter=6,
)
legenda = ParagraphStyle(
    "legenda", parent=body, fontSize=9.5, alignment=TA_CENTER,
    textColor=colors.HexColor("#444444"), spaceBefore=4,
)
capa_center = ParagraphStyle(
    "capa", parent=body, alignment=TA_CENTER, spaceAfter=14, fontSize=12,
)


def figura(arquivo, largura=LARGURA_UTIL * 0.82):
    """Imagem centralizada preservando proporção."""
    caminho = FIGURAS / arquivo
    iw, ih = ImageReader(str(caminho)).getSize()
    altura = largura * ih / iw
    img = Image(str(caminho), width=largura, height=altura)
    img.hAlign = "CENTER"
    return img


def p(texto):
    return Paragraph(texto, body)


# ---------------------------------------------------------------- dados reais
comp = pd.read_csv(ARTEFATOS / "comparativo.csv")
meta = json.loads((ARTEFATOS / "metadata.json").read_text(encoding="utf-8"))
melhor = meta["melhor_modelo"]

elements = []

# ---------------------------------------------------------------- CAPA
elements += [
    Spacer(1, 3 * cm),
    Paragraph("UNIVERSIDADE DO OESTE DE SANTA CATARINA – UNOESC",
              ParagraphStyle("u", parent=capa_center, fontName="Times-Bold", fontSize=14)),
    Spacer(1, 2 * cm),
    Paragraph("ANÁLISE E CLASSIFICAÇÃO DE QUALIDADE DE VINHOS UTILIZANDO "
              "MACHINE LEARNING E INTELIGÊNCIA ARTIFICIAL GENERATIVA",
              ParagraphStyle("t", parent=capa_center, fontName="Times-Bold", fontSize=15, leading=20)),
    Spacer(1, 3 * cm),
    Paragraph("Aluno: Fernando Germano Klein", capa_center),
    Paragraph("Disciplina: Inteligência Artificial e Sistemas Inteligentes", capa_center),
    Paragraph("Professor: Jacson Matte", capa_center),
    Spacer(1, 3 * cm),
    Paragraph("Chapecó – SC", capa_center),
    Paragraph("2026", capa_center),
    PageBreak(),
]

# ---------------------------------------------------------------- 1 INTRODUÇÃO
elements += [
    Paragraph("1 Introdução", h1),
    p("Este trabalho unifica os conceitos de exploração de dados, aprendizado de "
      "máquina e inteligência artificial generativa, com o objetivo de construir um "
      "<b>agente preditivo especialista</b>. A partir da base de dados <i>Wine "
      "Quality</i>, são treinados e comparados diferentes algoritmos de classificação "
      "e o modelo de melhor desempenho é integrado a uma interface web e a um agente "
      "inteligente, capaz não apenas de realizar a predição, mas também de traduzir e "
      "explicar o resultado em linguagem natural ao usuário final."),
    p("Em relação ao trabalho anterior, foram aplicados novos algoritmos — incluindo o "
      "Naive Bayes — e adicionadas as etapas de back-end, integração com um modelo de "
      "linguagem (Google Gemini) e construção de uma interface interativa."),
]

# ---------------------------------------------------------------- 2 DATASET
elements += [
    Paragraph("2 Descrição do Dataset", h1),
    p("Foi utilizado o dataset <i>Wine Quality</i> (vinho tinto), composto por 1.599 "
      "amostras e 11 variáveis físico-químicas: acidez fixa, acidez volátil, ácido "
      "cítrico, açúcar residual, cloretos, dióxido de enxofre livre, dióxido de enxofre "
      "total, densidade, pH, sulfatos e teor alcoólico."),
    p("A variável alvo representa a qualidade do vinho, originalmente variando de 0 a "
      "10. Para este trabalho, ela foi transformada em binária:"),
    p("• <b>1</b>: vinhos com qualidade maior ou igual a 7 (bons);<br/>"
      "• <b>0</b>: vinhos com qualidade menor que 7 (ruins)."),
]

# ---------------------------------------------------------------- 3 PRÉ-PROCESSAMENTO
elements += [
    Paragraph("3 Pré-processamento dos Dados", h1),
    p("Inicialmente foi realizada a verificação de valores nulos, não sendo "
      "identificadas inconsistências. Em seguida, as variáveis foram padronizadas com o "
      "método <i>StandardScaler</i>. Diferentemente do trabalho anterior, o "
      "padronizador foi ajustado <b>apenas sobre o conjunto de treino</b> e aplicado ao "
      "conjunto de teste, evitando o vazamento de informação (<i>data leakage</i>)."),
    p("Os dados foram divididos em treino (70%) e teste (30%) de forma "
      "<b>estratificada</b>, preservando a proporção entre as classes. A análise das "
      "distribuições evidenciou a presença de <i>outliers</i> em algumas variáveis; "
      "optou-se por mantê-los, pois representam características reais do dataset."),
]

# ---------------------------------------------------------------- 4 EDA
elements += [
    Paragraph("4 Análise Exploratória dos Dados", h1),
    Paragraph("4.1 Matriz de Correlação", h2),
    p("A matriz de correlação indicou que a variável <i>álcool</i> possui correlação "
      "positiva com a qualidade do vinho, enquanto a <i>acidez volátil</i> apresenta "
      "correlação negativa."),
    figura("correlacao.png"),
    Paragraph("Figura 1: Matriz de correlação", legenda),
    Paragraph("4.2 Boxplot", h2),
    p("Os boxplots evidenciaram a presença de <i>outliers</i> em diversas variáveis, "
      "principalmente no dióxido de enxofre total e no açúcar residual."),
    figura("boxplot.png"),
    Paragraph("Figura 2: Boxplot das variáveis", legenda),
    Paragraph("4.3 Distribuição da Qualidade", h2),
    p("Observa-se um desbalanceamento entre as classes, com predominância de vinhos de "
      "qualidade mais baixa (classe 0)."),
    figura("distribuicao.png", largura=LARGURA_UTIL * 0.6),
    Paragraph("Figura 3: Distribuição da qualidade", legenda),
]

# ---------------------------------------------------------------- 5 MODELOS
elements += [
    Paragraph("5 Modelos Utilizados", h1),
    p("Foram aplicados quatro algoritmos de classificação:"),
    p("• <b>Regressão Linear Múltipla</b>: modelo de regressão cuja saída contínua é "
      "convertida em classe por meio de um limiar de 0,5;<br/>"
      "• <b>KNN (K-Vizinhos Mais Próximos)</b>: classifica cada amostra com base na "
      "classe dos vizinhos mais próximos no espaço de atributos;<br/>"
      "• <b>MLP (Perceptron Multicamadas)</b>: rede neural artificial com camadas "
      "ocultas, capaz de modelar relações não lineares;<br/>"
      "• <b>Naive Bayes</b>: modelo probabilístico baseado no teorema de Bayes, "
      "assumindo independência entre as variáveis."),
]

# ---------------------------------------------------------------- 6 AJUSTE
elements += [
    Paragraph("6 Ajuste de Parâmetros", h1),
    p("Para os modelos com hiperparâmetros relevantes, utilizou-se a técnica de busca "
      "em grade (<i>GridSearchCV</i>) com validação cruzada de 5 dobras. No <b>KNN</b> "
      "foram testados o número de vizinhos e a forma de ponderação; no <b>MLP</b>, o "
      "tamanho das camadas ocultas e o termo de regularização (<i>alpha</i>). A "
      "Regressão Linear e o Naive Bayes foram treinados com sua configuração padrão. "
      "Esses ajustes contribuíram para melhorar o desempenho geral dos modelos."),
]

# ---------------------------------------------------------------- 7 MÉTRICAS
elements += [
    Paragraph("7 Métricas de Avaliação", h1),
    p("Foram utilizadas as seguintes métricas: <b>acurácia</b>, <b>precisão</b>, "
      "<b>sensibilidade</b> (<i>recall</i>) e <b>especificidade</b>. Adicionalmente, "
      "calculou-se o <b>F1-score</b> (média harmônica entre precisão e sensibilidade) "
      "como critério de desempate, por ser mais adequado a bases desbalanceadas."),
]

# ---------------------------------------------------------------- 8 RESULTADOS
tabela_dados = [["Modelo", "Acurácia", "Precisão", "Sensib.", "Especif.", "F1"]]
for _, r in comp.iterrows():
    tabela_dados.append([
        r["Modelo"], f"{r['Acurácia']:.3f}", f"{r['Precisão']:.3f}",
        f"{r['Sensibilidade']:.3f}", f"{r['Especificidade']:.3f}", f"{r['F1']:.3f}",
    ])

tabela = Table(tabela_dados, hAlign="CENTER")
tabela.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7b1e3b")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5eef0")]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))

elements += [
    Paragraph("8 Resultados", h1),
    tabela,
    Spacer(1, 4),
    Paragraph("Tabela 1: Comparação dos modelos", legenda),
    Spacer(1, 10),
    figura("comparativo.png", largura=LARGURA_UTIL * 0.7),
    Paragraph("Figura 4: Comparação de acurácia dos modelos", legenda),
]

# ---------------------------------------------------------------- 9 DISCUSSÃO
elements += [
    Paragraph("9 Discussão e Análise Crítica", h1),
    p(f"O modelo <b>{melhor}</b> apresentou o melhor equilíbrio geral, com o maior "
      "F1-score, combinando boa acurácia e um desempenho razoável na identificação da "
      "classe minoritária. A seguir, discutem-se as vantagens e limitações de cada "
      "algoritmo testado:"),
    p("• <b>Regressão Linear Múltipla</b> — <i>Vantagem:</i> simplicidade e alta "
      "especificidade. <i>Limitação:</i> por ser um modelo de regressão adaptado à "
      "classificação, obteve sensibilidade muito baixa (0,108), praticamente ignorando "
      "os vinhos bons; é a opção menos adequada para o problema."),
    p("• <b>KNN</b> — <i>Vantagem:</i> melhor F1-score e bom equilíbrio entre as "
      "métricas, sem premissas sobre a distribuição dos dados. <i>Limitação:</i> "
      "sensível à escala (mitigada pela padronização) e com custo de predição "
      "proporcional ao tamanho da base."),
    p("• <b>MLP</b> — <i>Vantagem:</i> capacidade de modelar relações não lineares, "
      "com desempenho próximo ao do KNN. <i>Limitação:</i> maior custo de treinamento, "
      "mais hiperparâmetros e risco de overfitting em bases pequenas."),
    p("• <b>Naive Bayes</b> — <i>Vantagem:</i> obteve a maior sensibilidade (0,692), "
      "sendo o melhor para detectar vinhos bons. <i>Limitação:</i> baixa precisão "
      "(muitos falsos positivos), decorrente da suposição de independência entre as "
      "variáveis, que não se sustenta neste dataset."),
    p("De modo geral, todos os modelos enfrentaram dificuldade com a classe "
      "minoritária, em razão do desbalanceamento dos dados — característica recorrente "
      "neste conjunto."),
]

# ---------------------------------------------------------------- 10 ARQUITETURA
elements += [
    Paragraph("10 Arquitetura do Agente Inteligente", h1),
    p("A solução final é composta por três camadas integradas:"),
    p("<b>1. Camada de Machine Learning.</b> O melhor modelo, juntamente com o "
      "padronizador e os metadados (ordem das variáveis e métricas), é exportado em "
      "disco (<i>joblib</i>). Isso desacopla o treinamento da execução: o modelo é "
      "treinado uma vez e reutilizado pela aplicação."),
    p("<b>2. Camada de Back-end e Agente.</b> Um servidor <b>FastAPI</b> expõe as "
      "rotas <font face='Courier'>/predict</font> (apenas a predição) e "
      "<font face='Courier'>/analisar</font> (predição + explicação). As 11 variáveis "
      "de entrada são validadas com <i>Pydantic</i>. Após a predição, o resultado e os "
      "valores informados são enviados ao modelo de linguagem <b>Google Gemini</b>, "
      "guiado por um <i>System Prompt</i> que o instrui a explicar o resultado em "
      "português, de forma fundamentada e <b>sem alucinar</b> — utilizando apenas a "
      "predição e os dados fornecidos. Para garantir robustez, há tratamento de erros, "
      "tempo-limite e um modo de contingência que mantém a aplicação funcional mesmo "
      "sem acesso ao agente."),
    p("<b>3. Camada de Front-end.</b> Uma interface em <b>Streamlit</b> permite que o "
      "usuário informe as propriedades do vinho (ou carregue exemplos por classe) e "
      "visualize tanto o resultado bruto do modelo quanto a explicação gerada pelo "
      "agente, além dos gráficos exploratórios e da comparação entre os modelos."),
    p("<b>Fluxo resumido:</b> interface (Streamlit) → requisição à API (FastAPI) → "
      "modelo treinado gera a predição → agente (Gemini) traduz o resultado em "
      "linguagem natural → resposta exibida ao usuário."),
]

# ---------------------------------------------------------------- 11 CONCLUSÃO
elements += [
    Paragraph("11 Conclusão", h1),
    p(f"Conclui-se que o modelo <b>{melhor}</b> foi o mais eficiente para o problema "
      "proposto, considerando o equilíbrio entre as métricas avaliadas. A integração "
      "entre o modelo de aprendizado de máquina, a API e o agente de inteligência "
      "artificial generativa mostrou-se eficaz, permitindo que o resultado matemático "
      "fosse traduzido em uma explicação clara e acessível ao usuário final."),
    p("Como trabalhos futuros, sugerem-se técnicas de balanceamento de dados (como o "
      "<i>SMOTE</i>) para melhorar a detecção da classe minoritária, além de um ajuste "
      "mais amplo de hiperparâmetros e a avaliação de modelos de <i>ensemble</i>."),
]

# ---------------------------------------------------------------- build
doc = SimpleDocTemplate(
    str(SAIDA), pagesize=A4,
    leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    title="Relatório Técnico - Classificação de Qualidade de Vinhos",
    author="Fernando Germano Klein",
)
doc.build(elements)
print(f"PDF gerado em: {SAIDA}")
