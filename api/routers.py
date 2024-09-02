from fastapi import APIRouter, UploadFile

from llm_retrieval_qa.splitter import split_html
from llm_retrieval_qa.configs import model_config, get_embedding_fn, vector_store_config


router = APIRouter()

embedding_fn = get_embedding_fn(model_config["embedding_cfgs"])


@router.post("/upload_doc/")
async def upload_doc(
    file: UploadFile,
    chunk_size: int = 500,
    chunk_overlap: int = 30,
    vector_store_type: str = 'faiss',
):
    doc_fname = file.filename
    contents = await file.read()
    contents = contents.decode()

    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
        ("h5", "Header 5"),
        ("table", 'table'),
    ]

    splits = split_html(
        contents,
        encoding='utf-8',
        sections_to_split=headers_to_split_on,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ",", "."],
    )

    # store data to vector database
    if vector_store_type == "milvus":
        from llm_retrieval_qa.vector_store.milvus import DbMilvus

        vector_db = DbMilvus(
            embedding_fn,
            vector_store_config.uri,
            db_name=vector_store_config.db_name,
            collection_name=vector_store_config.collection,
        )

        texts = [x.dict()["page_content"] for x in splits]
        # check if the doc is already in the vector db using doc_fname
        exist_docs = vector_db.get(f'doc_fname == "{doc_fname}"')
        if len(exist_docs) == 0:
            _ = vector_db.create(texts, doc_fname=doc_fname)
    elif vector_store_type == "faiss":
        from llm_retrieval_qa.vector_store.faiss import DbFAISS

        vector_db = DbFAISS(embedding_fn, vector_store_config.uri, normalize=True)

        if not vector_db.check_by_doc_fname(doc_fname) or doc_fname == "default":
            texts = [x.dict()["page_content"] for x in splits]
            _ = vector_db.create(texts, doc_fname=doc_fname)
            vector_db.save_local(vector_store_config.uri)
    else:
        vector_db = None
    return {"documents": splits}
