from datasets import Dataset

from card4a_query_llm_lcel import ask_rag_chain, build_rag_chain
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)


questions = [
    # "Quanto o cliente Levi Fogaça gastou em 5 de abril de 2023?", -> Total diário Levi Fogaça 5/4/2023
    #          "Qual o total mensal que o cliente Levi Fogaça gastou em abril de 2023?", -> Total mensal Levi Fogaça abril 2023
             # "Quanto o cliente Levi Fogaça gastou em 2023?", -> Total anual Levi Fogaça 2023
             # "Quanto o cliente Levi Fogaça gastou em abril, maio e junho de 2023?", -> Soma total mensal Levi Fogaça abril maio junho 2023
             # "Quanto o cliente Levi Fogaça gastou em abril comparado com junho de 2023?", -> Comparativo em porcentagem total mensal Levi Fogaça entre abril junho 2023
             # "Qual o total transacionado na carteira em 5 de abril de 2023?", -> Total diário 05/04/2023
             # "Qual o total mensal transacionado na carteira em abril de 2023?", -> Total mensal carteira abril 2023
             # "Qual o total anual transacionado na carteira em 2023?", -> Total anual carteira 2023
             # "Qual o total transacionado na carteira em abril, maio e junho de 2023?", -> mensal abril maio junho 2023 soma
             # "Qual o total transacionado na carteira em abril comparado com junho de 2023?", -> Comparativo em porcentagem total mensal carteira entre abril junho 2023
             # "Qual o total transacionado no cartão de crédito em 5 de abril de 2023?", -> Só tem crédito no DB
             # "Qual o total transacionado no cartão de crédito em abril de 2023?",-> Só tem crédito no DB
             # "Qual o total anual transacionado no cartão de crédito em 2023?",-> Só tem crédito no DB
             # "Qual o total mensal transacionado no cartão de crédito em abril, maio e junho de 2023?",-> Só tem crédito no DB
             # "Qual o total transacionado no cartão de crédito em abril comparado com junho de 2023?",-> Só tem crédito no DB
             # "Qual o total transacionado no cartão de crédito BLACK em 5 de abril de 2023?", -> só tem as contagens no DB
             # "Qual o total transacionado no cartão de crédito BLACK em abril de 2023?",-> só tem as contagens no DB
             # "Qual o total transacionado no cartão de crédito BLACK em abril, maio e junho de 2023?",-> só tem as contagens no DB
             # "Qual o percentual de evolução do valor total mensal transacionado no cartão de crédito BLACK em abril comparado com junho de 2023?",
             # "Sumarize a evolução do valor total mensal dos gastos do cliente Levi Fogaça nos últimos 6 meses (julho, agosto, setembro, outubro, novembro e dezembro de 2023)?", -> Total mensal Levi Fogaça julho agosto setembro outubro novembro dezembro 2023
             # "Calcule o valor médio mensal transacionado no mês de abril e compare com o valor médio mensal dos gastos do cliente Levi Fogaça no mesmo mês. -> média mensal carteira abril 2023 média mensal Levi Fogaça abril 2023. Comparativo em porcentagem
                # Compare as duas médias e me dê uma resposta em porcentagem ?",
            ]
ground_truth = [
    # ["O cliente Levi Fogaça não realizou nenhuma transação em 5 de abril de 2023"],
    # ["R$ 65244.69"],
    # ["R$ 769380.30"],
    # ["R$ 208072.21"],
    # ["O cliente Levi Fogaça gastou R$ 65244.69 em abril e R$ 64056.85 em junho de 2023 (-1.8%)."],
    # ["R$ 16193.31"],
    # ["R$ 612354.40"],
    # ["R$ R$ 7558624.98"],
    # ["R$ 1999603.18"],
    # ["O cliente Levi Fogaça gastou R$ 612354.40 em abril e R$ 652198.30 em junho de 2023 (+6.5%)."],
    # ["R$ 16193.31"],
    # ["R$ R$ 612354.40"],
    # ["R$ 7558624.98"],
    # ["R$ 1999603.18"],
    # ["O cliente Levi Fogaça gastou R$ 612354.40 em abril e R$ 652198.30 em junho de 2023 (+6.5%)."],
    # ["R$ 3546.29"],
    # ["R$ 106070.97"],
    # ["R$ 427014.73"],
    # ["O cliente Levi Fogaça gastou R$ 106070.97 em abril e R$ 134678.39 em junho de 2023 (+27.0%)."],
    # ["07/2023: R$ 71482.57, 08/2023: R$ 71263.53, 09/2023: R$ 80779.93, 10/2023: R$ 84565.67, 11/2023: R$ 41154.85, 12/2023: R$ 68775.89"],
    # ["O cliente Levi Fogaça gastou em média +14.0% (R$ 2836.73 em abril de 2023) em relação à média de todos os clientes de R$ 2489.25."]
                 ]
answers = []
contexts = []
# retriever, _, _, _ = build_rag_chain()          #lcel
retriever, chain = build_rag_chain()
# Inference
for query in questions:
  answers.append(ask_rag_chain(query, chain))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truth
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

if __name__ == '__main__':
    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    df = result.to_pandas()
    print(df)
