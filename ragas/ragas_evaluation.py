from datasets import Dataset

from card4a_query_llm_lcel import ask_rag_chain, build_rag_chain
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)


questions = ["Qual o valor total de transações do cliente Carlos Eduardo Rodrigues, no débito, em março de 2023?",
             "Qual o valor total de transações do cliente Carlos Eduardo Rodrigues em março de 2023?"
            ]
ground_truths = [["O valor total gasto no debito pelo cliente Carlos Eduardo Rodrigues no mês de março de 2023 foi de R$ 62126.99"],
                ["O valor total gasto pelo cliente Carlos Eduardo Rodrigues no mês de março de 2023 foi de R$ 129840.87, "
                 "sendo R$ 62126.99 no debito e R$ 67713.88 no credito."]]
answers = []
contexts = []
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

    pd.set_option("display.max_colwidth", None)
    df = result.to_pandas()
    df.to_csv('results.csv')
