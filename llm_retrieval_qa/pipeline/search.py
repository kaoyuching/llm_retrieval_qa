

def similarity_search_faiss(vector_db, question, top_k: int = 10, threshold=None):
    query_docs = vector_db.similarity_search_with_score(
        question,
        k=top_k,
    )

    doc_str = []
    contexts = []
    scores = []
    for doc in query_docs:
        if threshold is None or _score <= threshold:
            doc_str.append(doc["text"])
            contexts.append(doc["text"])
        scores.append(doc["distance"])
    return doc_str, contexts, scores


def similarity_search_milvus(vector_db, question, top_k: int = 10, threshold=None):
    query_docs = vector_db.similarity_search_with_score(
        question,
        top_k=top_k,
    )

    doc_str = []
    contexts = []
    scores = []
    for doc in query_docs:
        _score = doc["distance"]
        if threshold is None or _score <= threshold:
            doc_str.append(doc["entity"]["text"])
            contexts.append(doc["entity"]["text"])
        scores.append(_score)
    return doc_str, contexts, scores


def similarity_search(vector_db, question, top_k: int = 10, threshold=None):
    if vector_db.__class__.__name__ == "DbMilvus":
        doc_str, contexts, scores = similarity_search_milvus(vector_db, question, top_k=top_k)
    else:
        doc_str, contexts, scores = similarity_search_faiss(vector_db, question, top_k=top_k)
    return doc_str, contexts, scores
