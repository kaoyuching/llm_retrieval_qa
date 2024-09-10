from llm_retrieval_qa.pipeline.reranking import ReRanking


def similarity_search_faiss(vector_db, question, top_k: int = 10, threshold=None, extend_num: int = 0):
    # query_docs: List[Dict(id, distance, text)]
    contexts = []
    search_ids = []
    query_docs = vector_db.similarity_search_with_score(
        question,
        k=top_k,
    )

    if len(query_docs) == 0:
        return contexts, search_ids

    for doc in query_docs:
        _score = doc["distance"]
        _id = doc["id"]
        if threshold is None or _score <= threshold:
            range_id = range(max(0, _id - extend_num), min(_id + (extend_num + 1), vector_db.ntotal))
            search_ids.extend(list(range_id))

    search_ids = set(sorted(search_ids))

    texts = [vector_db.get_by_id(i)['text'] for i in search_ids]
    return texts, search_ids


def similarity_search_milvus(vector_db, question, top_k: int = 10, threshold=None, extend_num: int = 0):
    contexts = []
    search_ids = []
    # query_docs: List[Dict(id, distance, entity[text, doc_fname, doc_id])]
    query_docs = vector_db.similarity_search_with_score(
        question,
        top_k=top_k,
    )

    if len(query_docs) == 0:
        return contexts, search_ids

    for doc in query_docs:
        _score = doc["distance"]
        _id = doc["id"]
        _doc_id = doc["entity"]["doc_id"]
        _doc_fname = doc["entity"]["doc_fname"]
        if threshold is None or _score <= threshold:
            range_id = range(max(0, _doc_id - extend_num), min(_doc_id + (extend_num + 1), vector_db.doc_ntotal(_doc_fname)))
            range_id = [vector_db.get(filter=f"doc_id=={i} and doc_fname=='{_doc_fname}'")[0]["id"] for i in range_id]  # vector db id
            search_ids.extend(range_id)

    search_ids = set(sorted(search_ids))

    texts = [vector_db.get_by_id(i)['text'] for i in search_ids]
    return texts, search_ids


def similarity_search(
    vector_db,
    question,
    top_k: int = 10,
    threshold=None,
    extend_num: int = 0,
    reranking: bool = False,
    rerank_topk: int = 5,
):
    if rerank_topk > top_k:
        rerank_topk = top_k

    if vector_db.__class__.__name__ == "DbMilvus":
        contexts, search_ids = similarity_search_milvus(vector_db, question, top_k=top_k, threshold=threshold, extend_num=extend_num)
    else:
        contexts, search_ids = similarity_search_faiss(vector_db, question, top_k=top_k, threshold=threshold, extend_num=extend_num)

    # TODO: reranking
    # reranking: https://www.rungalileo.io/blog/mastering-rag-how-to-select-a-reranking-model
    _tmp = {x: i for x, i in zip(contexts, search_ids)}
    if reranking:
        reranking = ReRanking()
        res = reranking.rank([question], contexts)[:rerank_topk]
        contexts = [x["text"] for x in res]
        search_ids = [_tmp[x] for x in contexts]
    print(contexts, flush=True)
    print(len(contexts), flush=True)
    return contexts, search_ids
