"""
Etapa B - Agente inteligente (Gemini).

Recebe a predição do modelo + os valores informados e pede ao Gemini uma
explicação em linguagem natural, fundamentada e sem alucinação.
"""

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

SYSTEM_PROMPT = """Você é um enólogo assistente que explica, em português do Brasil,
o resultado de um modelo de machine learning que classifica vinhos tintos como
BOM ou RUIM a partir de 11 propriedades físico-químicas.

Regras:
- Baseie-se EXCLUSIVAMENTE na predição do modelo e nos valores numéricos informados.
  Não invente dados, marcas, safras, preços ou informações que não foram fornecidas.
- Deixe claro que é uma estimativa estatística, não um laudo definitivo.
- Comente de forma objetiva quais variáveis informadas tendem a puxar o resultado
  para BOM ou RUIM, usando relações conhecidas na base Wine Quality:
  * álcool mais alto e acidez volátil mais baixa tendem a indicar vinhos melhores;
  * sulfatos e acidez cítrica costumam ter relação positiva com a qualidade.
- Seja claro e didático, em 2 a 4 parágrafos curtos. Não use jargão excessivo.
"""


def _gerar_offline(classe, probabilidade, dados):
    """Fallback usado quando a API do Gemini não está disponível, para a aplicação
    continuar funcionando (a predição do modelo nunca depende do LLM)."""
    rotulo = "BOM" if classe == 1 else "RUIM"
    return (
        f"[Explicação automática — agente de IA indisponível no momento]\n\n"
        f"O modelo classificou este vinho como **{rotulo}** com probabilidade "
        f"estimada de {probabilidade:.0%}. Os valores de maior influência costumam "
        f"ser o teor alcoólico ({dados.get('alcohol')}) e a acidez volátil "
        f"({dados.get('volatile acidity')}): álcool mais alto e acidez volátil mais "
        f"baixa tendem a favorecer um resultado melhor. Trata-se de uma estimativa "
        f"estatística, não de um laudo definitivo."
    )


def explicar_predicao(classe, probabilidade, dados):
    """Gera a explicação em linguagem natural. Retorna um dict com o texto e a
    'fonte' (qual mecanismo respondeu), para a interface poder evidenciar se a
    resposta veio do Gemini ou do modo de contingência. Nunca lança exceção."""
    rotulo = "BOM" if classe == 1 else "RUIM"

    if not GEMINI_API_KEY:
        return {
            "texto": _gerar_offline(classe, probabilidade, dados),
            "fonte": "Contingência (sem chave de API configurada)",
        }

    try:
        from google import genai
        from google.genai import types

        cliente = genai.Client(api_key=GEMINI_API_KEY)

        atributos = "\n".join(f"- {k}: {v}" for k, v in dados.items())
        prompt = (
            f"Resultado do modelo: {rotulo} "
            f"(probabilidade de ser bom: {probabilidade:.2%}).\n\n"
            f"Valores físico-químicos informados:\n{atributos}\n\n"
            f"Explique este resultado ao usuário final seguindo as regras."
        )

        resposta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        texto = (resposta.text or "").strip()
        if texto:
            return {"texto": texto, "fonte": f"Google Gemini · {GEMINI_MODEL}"}
        return {
            "texto": _gerar_offline(classe, probabilidade, dados),
            "fonte": "Contingência (resposta vazia do Gemini)",
        }
    except Exception as exc:  # noqa: BLE001 - fallback intencional e abrangente
        print(f"[agente] Falha ao chamar o Gemini: {exc}")
        return {
            "texto": _gerar_offline(classe, probabilidade, dados),
            "fonte": "Contingência (falha ao acessar o Gemini)",
        }
