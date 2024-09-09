

def similarity_search_faiss(vector_db, question, top_k: int = 10, threshold=None, extend_num: int = 0):
    # query_docs: List[Dict(id, distance, text)]
    query_docs = vector_db.similarity_search_with_score(
        question,
        k=top_k,
    )

    doc_str = []
    contexts = []
    scores = []
    search_ids = []
    for doc in query_docs:
        _score = doc["distance"]
        _id = doc["id"]
        if threshold is None or _score <= threshold:
            range_id = range(max(0, _id - extend_num), min(_id + (extend_num + 1), vector_db.ntotal))
            search_ids.extend(list(range_id))
        scores.append(_score)

    search_ids = set(sorted(search_ids))

    text = '\n'.join([vector_db.get_by_id(i)['text'] for i in search_ids])
    doc_str.append(text)
    contexts.append(text)
    return doc_str, contexts, scores


def similarity_search_milvus(vector_db, question, top_k: int = 10, threshold=None, extend_num: int = 0):
    doc_str = []
    contexts = []
    scores = []
    # query_docs: List[Dict(id, distance, entity[text, doc_fname, doc_id])]
    query_docs = vector_db.similarity_search_with_score(
        question,
        top_k=top_k,
    )

    if len(query_docs) == 0:
        return doc_str, contexts, scores

    search_ids = []
    for doc in query_docs:
        _score = doc["distance"]
        _id = doc["id"]
        _doc_id = doc["entity"]["doc_id"]
        _doc_fname = doc["entity"]["doc_fname"]
        if threshold is None or _score <= threshold:
            range_id = range(max(0, _doc_id - extend_num), min(_doc_id + (extend_num + 1), vector_db.doc_ntotal(_doc_fname)))
            range_id = [vector_db.get(filter=f"doc_id=={i} and doc_fname=='{_doc_fname}'")[0]["id"] for i in range_id]
            search_ids.extend(range_id)
        scores.append(_score)

    search_ids = set(sorted(search_ids))

    text = '\n'.join([vector_db.get_by_id(i)['text'] for i in search_ids])
    doc_str.append(text)
    contexts.append(text)
    return doc_str, contexts, scores


def similarity_search(vector_db, question, top_k: int = 10, threshold=None, extend_num: int = 0):
    if vector_db.__class__.__name__ == "DbMilvus":
        doc_str, contexts, scores = similarity_search_milvus(vector_db, question, top_k=top_k, threshold=threshold, extend_num=extend_num)
    else:
        doc_str, contexts, scores = similarity_search_faiss(vector_db, question, top_k=top_k, threshold=threshold, extend_num=extend_num)
    print(doc_str)
    return doc_str, contexts, scores
