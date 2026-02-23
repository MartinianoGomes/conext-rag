import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings


def run():

    with open("eval_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)

    base_llm = ChatOllama(
        model="mistral",
        temperature=0,
    )

    llm = LangchainLLMWrapper(base_llm)

    base_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = LangchainEmbeddingsWrapper(base_embeddings)

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm,
        embeddings=embeddings
    )

    print(result)


if __name__ == "__main__":
    run()