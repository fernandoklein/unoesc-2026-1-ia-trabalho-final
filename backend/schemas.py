"""Modelos de entrada/saída da API (validação via Pydantic)."""

from pydantic import BaseModel, Field


class EntradaVinho(BaseModel):
    """As 11 variáveis físico-químicas esperadas pelo modelo.

    Os aliases batem com os nomes das colunas do dataset (com espaços), de modo
    que o front pode enviar tanto o nome com underscore quanto o original.
    Faixas (ge/le) cobrem com folga os valores observados no dataset."""

    fixed_acidity: float = Field(..., ge=0, le=20, alias="fixed acidity")
    volatile_acidity: float = Field(..., ge=0, le=2, alias="volatile acidity")
    citric_acid: float = Field(..., ge=0, le=1.5, alias="citric acid")
    residual_sugar: float = Field(..., ge=0, le=20, alias="residual sugar")
    chlorides: float = Field(..., ge=0, le=1)
    free_sulfur_dioxide: float = Field(..., ge=0, le=100, alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., ge=0, le=300, alias="total sulfur dioxide")
    density: float = Field(..., ge=0.9, le=1.1)
    pH: float = Field(..., ge=2, le=5)
    sulphates: float = Field(..., ge=0, le=2)
    alcohol: float = Field(..., ge=5, le=20)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "fixed acidity": 7.4,
                "volatile acidity": 0.7,
                "citric acid": 0.0,
                "residual sugar": 1.9,
                "chlorides": 0.076,
                "free sulfur dioxide": 11,
                "total sulfur dioxide": 34,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        },
    }


class RespostaPredicao(BaseModel):
    classe: int
    rotulo: str
    probabilidade: float
    modelo: str


class RespostaAnalise(RespostaPredicao):
    explicacao: str
