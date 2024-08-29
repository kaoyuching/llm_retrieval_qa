from typing import List
import numpy as np
import faiss
import uuid
import dill

from llm_retrieval_qa.embeddings.huggingface import HFEmbedding


model_name = "GanymedeNil/text2vec-large-chinese"
hf_embedding = HFEmbedding(model_name, normalize_embeddings=True)
texts = ['This is an apple.', 'LLM stands for large language model.']
vectors = hf_embedding.embedding(texts)
text_embedding_pairs = zip(texts, vectors)

# create a search vector
vector_dim = hf_embedding.embedding_dim
index = faiss.IndexFlatL2(vector_dim)

# load vector db
# with open("./example_db/faiss_doc_store.pkl", "rb") as f:
    # restore_index, restore_doc_store, restore_index_to_docid = dill.load(f)
    # restore_doc_ids = restore_doc_store.keys()
# index.merge_from(restore_index)
init_db_size = index.ntotal

# if normalize
faiss.normalize_L2(vectors)
index.add(vectors)

# doc and index mapping
def gen_uuid(n, exist_ids: List = []):
    if len(exist_ids) == 0:
        return [str(uuid.uuid4()) for _ in texts]

    doc_ids = []
    for _ in range(n):
        is_duplicate = True
        while is_duplicate:
            idx = str(uuid.uuid4())
            is_duplicate = idx in exist_ids
            if not is_duplicate:
                doc_ids.append(idx)
                break
    return doc_ids

doc_ids = gen_uuid(len(texts), restore_doc_ids)
doc_store = {i: text for i, text in zip(doc_ids, texts)}
index_to_docid = {(i + init_db_size): _id for i, _id in enumerate(doc_ids)}
doc_store.update(restore_doc_store)
index_to_docid.update(restore_index_to_docid)

# search example
search_text = ["Where is apple?"]
search_vectors = hf_embedding.embedding(search_text)
faiss.normalize_L2(search_vectors)

k = index.ntotal
D, I = index.search(search_vectors, k=k)  # return Distance, Index
print(D, I)

# get from id
for idx in I:
    for i in idx:
        docid = index_to_docid[i]
        search_text = doc_store[docid]
        print('search result:', docid, search_text)


# delete data in vector_db
index.remove_ids(np.array([1]))

# save the results
# faiss.write_index(index, "./example_db/doc_index.faiss")
# with open("./example_db/faiss_doc_store.pkl", "wb") as f:
    # dill.dump((index, doc_store, index_to_docid), f)
