import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests


def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def run():
    if not check_ollama():
        print("Ollama desligado. Rode 'ollama serve' no terminal para iniciar o serviço.")
        return

    print("📂 Carregando eval_data.json...")

    with open("../eval_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)

    print("🤖 Inicializando LLM para avaliação...")
    llm = ChatOllama(
        model="llama3",
        timeout=120,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📊 Rodando RAGAS (isso pode demorar alguns minutos)...")

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=True,
    )

    print("\n📈 Resultado Final:")
    print(result)


if __name__ == "__main__":
    run()
