from datasets import Dataset

from chains.llm_lcel_advanced import ask_rag_chain, build_rag_chain
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)


questions = ["Quanto o cliente Levi Fogaça gastou em 5 de abril de 2023?",
             # "Quanto o cliente Levi Fogaça gastou em abril de 2023?",
             # "Quanto o cliente Levi Fogaça gastou em 2023?",
             # "Quanto o cliente Levi Fogaça gastou em abril, maio e junho de 2023?",
             # "Quanto o cliente Levi Fogaça gastou em abril comparado com junho de 2023?",
             # "Qual o total transacionado na carteira em 5 de abril de 2023?",
             "Qual o total transacionado na carteira em abril de 2023?",
             # "Qual o total transacionado na carteira em 2023?",
             # "Qual o total transacionado na carteira em abril, maio e junho de 2023?",
             # "Qual o total transacionado na carteira em abril comparado com junho de 2023?",
             # "Qual o total transacionado no cartão de crédito em 5 de abril de 2023?",
             # "Qual o total transacionado no cartão de crédito em abril de 2023?",
             "Qual o total transacionado no cartão de crédito em 2023?",
             # "Qual o total transacionado no cartão de crédito em abril, maio e junho de 2023?",
             # "Qual o total transacionado no cartão de crédito em abril comparado com junho de 2023?",
             # "Qual o total transacionado no cartão de crédito BLACK em 5 de abril de 2023?",
             # "Qual o total transacionado no cartão de crédito BLACK em abril de 2023?",
             # "Qual o total transacionado no cartão de crédito BLACK em abril, maio e junho de 2023?",
             # "Qual o total transacionado no cartão de crédito BLACK em abril comparado com junho de 2023?",
             # "Qual a evolução mensal dos gastos do cliente Levi Fogaça nos últimos 6 meses (julho, agosto, setembro, outubro, novembro e dezembro de 2023)?",
             # "Em relação à média de gastos de todos os clientes em abril, como a cliente Levi Fogaça se compara?"
            ]
ground_truths = [
    ["O cliente Levi Fogaça não realizou nenhuma transação em 5 de abril de 2023"],
    # ["R$ 65244.69"],
    # ["R$ 769380.30"],
    # ["R$ 208072.21"],
    # ["O cliente Levi Fogaça gastou R$ 65244.69 em abril e R$ 64056.85 em junho de 2023 (-1.8%)."],
    # ["R$ 16193.31"],
    ["R$ 612354.40"],
    # ["R$ R$ 7558624.98"],
    # ["R$ 1999603.18"],
    # ["O cliente Levi Fogaça gastou R$ 612354.40 em abril e R$ 652198.30 em junho de 2023 (+6.5%)."],
    # ["R$ 16193.31"],
    # ["R$ R$ 612354.40"],
    ["R$ 7558624.98"],
    # ["R$ 1999603.18"],
    # ["O cliente Levi Fogaça gastou R$ 612354.40 em abril e R$ 652198.30 em junho de 2023 (+6.5%)."],
    # ["R$ 3546.29"],
    # ["R$ 106070.97"],
    # ["R$ 427014.73"],
    # ["O cliente Levi Fogaça gastou R$ 106070.97 em abril e R$ 134678.39 em junho de 2023 (+27.0%)."],
    # ["Levi Fogaça (ID 000000009): 07/2023: R$ 71482.57, 08/2023: R$ 71263.53, 09/2023: R$ 80779.93, 10/2023: R$ 84565.67, 11/2023: R$ 41154.85, 12/2023: R$ 68775.89"],
    # ["O cliente Levi Fogaça gastou em média R$ 2836.73 em abril de 2023 +14.0%) em relação à média de todos os clientes de R$ 2489.25."]
                 ]
answers = []
contexts = []
# retriever, _, _, _ = build_rag_chain()          #lcel
retriever, _, _, _ = build_rag_chain()
# Inference
for query in questions:
  answers.append(ask_rag_chain(query))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
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
