import sys
import os
import dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chat_models import init_chat_model
from transformers import logging
from rag_pipeline import (
    build_vectorstore,
    # create_llm,
    generate_questions,
    answer_question
)
from export_eval_data import save_eval_data

logging.set_verbosity_error()

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1")
os.environ["HUGGINGFACEAPI_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-sael")

def load_document(path: str):
    file_path = Path(path)

    if file_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_path.suffix.lower() == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError("Formato não suportado. Use .pdf ou .txt")

    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])


def run(document_path: str):
    document = load_document(document_path)

    vectorstore = build_vectorstore(document)
    retriever = vectorstore.as_retriever()

    model = init_chat_model(
        model=os.getenv("OPEN_MODEL", "openai-gpt-oss-120b"),
        model_provider="openai",
        base_url=os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1")
    )

    questions = generate_questions(model, document, n_questions=3)

    qa_pairs = []

    for question in questions:
        answer, docs = answer_question(model, retriever, question)

        qa_pairs.append({
            "question": question,
            "answer": answer,
            "contexts": [doc.page_content for doc in docs],
            "ground_truth": answer
        })

    save_eval_data(qa_pairs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/auto_context_rag.py caminho_do_documento.pdf")
    else:
        run(sys.argv[1])
