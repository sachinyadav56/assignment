from sentence_transformers import SentenceTransformer
import chromadb
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def load_documents():
    docs = []

    files = [
        ("data/policy_v1_2020.txt", {"status": "old", "version": 1, "year": 2020}),
        ("data/policy_v2_2023.txt", {"status": "active", "version": 2, "year": 2023}),
        ("data/policy_future_DRAFT_v3.txt", {"status": "draft", "version": 3, "year": 2026}),
    ]

    for path, meta in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append({
                "id": path,
                "content": text,
                "metadata": {**meta, "source": path}
            })

    return docs


def build_vector_db(documents):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    collection = client.create_collection(name="policies")

    for doc in documents:
        embedding = model.encode(doc["content"]).tolist()
        collection.add(
            documents=[doc["content"]],
            metadatas=[doc["metadata"]],
            ids=[doc["id"]],
            embeddings=[embedding]
        )

    return collection, model


def search(collection, model, query, k=3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results


def decide_policy(question: str):
    q = question.lower()
    if "plan" in q or "future" in q or "change" in q:
        return "include_draft"
    else:
        return "active_only"


def filter_results(results, policy_mode):
    filtered = []

    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    for meta, doc in zip(metadatas, documents):
        status = meta["status"]

        if policy_mode == "active_only":
            if status == "active":
                filtered.append((meta, doc))
        else:  # include_draft
            if status in ["active", "draft"]:
                filtered.append((meta, doc))

    return filtered


def generate_answer(question, filtered_docs):
    if not filtered_docs:
        return "No relevant policy found.", []

    # Build context
    context_parts = []
    sources = []

    for meta, doc in filtered_docs:
        context_parts.append(doc)
        sources.append(meta["source"])

    context = "\n\n".join(context_parts)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    prompt = f"""
You are an HR policy assistant.

Context:
{context}

Question: {question}

Rules:
- If draft policy is mentioned, clearly say it is not yet effective.
- Answer clearly and concisely.
- Base your answer only on the context.
"""

    response = llm.invoke(prompt)

    return response.content, list(set(sources))


if __name__ == "__main__":
    documents = load_documents()
    collection, model = build_vector_db(documents)

    questions = [
        "How many days can I work remotely?",
        "What is the meal allowance?",
        "Are there any plans to change the vacation policy?"
    ]

    for question in questions:
        print("\n==============================")
        print("Question:", question)

        results = search(collection, model, question, k=3)
        mode = decide_policy(question)
        filtered = filter_results(results, mode)

        answer, sources = generate_answer(question, filtered)

        print("Answer:", answer)
        print("Source(s):", ", ".join(sources))
