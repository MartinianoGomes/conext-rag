from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def build_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.split_documents([Document(page_content=text)])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


def create_llm(model_name="llama3"):
    return ChatOllama(model=model_name)


def generate_questions(llm, context, n_questions=3):
    prompt = f"""
    Gere {n_questions} perguntas relevantes baseadas no contexto abaixo.
    Responda apenas com as perguntas numeradas.

    Contexto:
    {context}
    """

    response = llm.invoke(prompt).content

    questions = []
    for line in response.split("\n"):
        line = line.strip()
        if line and any(char.isdigit() for char in line[:2]):
            question = line.split(".", 1)[-1].strip()
            questions.append(question)

    return questions


def answer_question(llm, retriever, question):
    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Responda à pergunta usando apenas o contexto abaixo.

    Contexto:
    {context}

    Pergunta:
    {question}
    """

    response = llm.invoke(prompt).content

    return response, docs
