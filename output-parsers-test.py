from env import OPENAI_API_KEY, OPENAI_MODEL_NAME

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL_NAME, temperature=0.7)


def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Conte uma piada sobre o seguinte assunto"),
        ("human", "{input}")
    ])

    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain.invoke({
        "input": "padre"
    })


def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Gere uma lista de 5 sinônimos para a palavra a seguir. "
         "Retorne os resultados como uma lista separada por vírgulas."),
        ("human", "{input}")
    ])

    parser = CommaSeparatedListOutputParser()

    chain = prompt | model | parser

    return chain.invoke({
        "input": "tristeza"
    })


def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extraia informações da frase a seguir.\nInstruções de formatação: {format_instructions}"),
        ("human", "{phrase}")
    ])

    class Person(BaseModel):
        recipe: str = Field(description="Nome da receita")
        ingredients: list = Field(description="Ingredientes")

    parser = JsonOutputParser(pydantic_object=Person)

    chain = prompt | model | parser

    return chain.invoke({
        "phrase": "Os ingredientes para uma pizza Margherita são tomate, cebola, queijo, manjericão",
        "format_instructions": parser.get_format_instructions()
    })


if __name__ == "__main__":
    print(call_json_output_parser())
