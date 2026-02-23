import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from rag_pipeline import (
    build_vectorstore,
    create_llm,
    generate_questions,
    answer_question
)
from export_eval_data import save_eval_data


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
    print("📄 Carregando documento...")
    text = load_document(document_path)

    print("🔎 Construindo vetor...")
    vectorstore = build_vectorstore(text)
    retriever = vectorstore.as_retriever()

    print("🤖 Inicializando LLM...")
    llm = create_llm("llama3")

    print("🧠 Gerando perguntas automaticamente...")
    questions = generate_questions(llm, text, n_questions=3)

    print(f"✔️ {len(questions)} perguntas geradas.")

    qa_pairs = []

    for question in questions:
        answer, docs = answer_question(llm, retriever, question)

        qa_pairs.append({
            "question": question,
            "answer": answer,
            "contexts": [doc.page_content for doc in docs],
            "ground_truth": answer
        })

    print("💾 Salvando dados para avaliação...")
    save_eval_data(qa_pairs)

    print("✅ Pipeline concluído.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/auto_context_rag.py caminho_do_documento.pdf")
    else:
        run(sys.argv[1])
