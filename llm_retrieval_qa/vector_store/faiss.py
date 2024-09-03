from typing import Set, List, Dict, Union, Optional
import os
import numpy as np
import faiss
import uuid
import dill


class DbFAISS():
    r"""
    faiss example

    model_name = "GanymedeNil/text2vec-large-chinese"
    hf_embedding = HFEmbedding(model_name, normalize_embeddings=True)
    vectors = hf_embedding.embedding(...)

    # create a search vector
    vector_dim = vectors.shape[1]
    index = faiss.IndexFlatL2(vector_dim)
    # if normalize
    faiss.normalize_L2(vectors)
    index.add(vectors)

    # search
    search_text = ...
    search_vectors = hf_embedding.embedding(search_text)
    faiss.normalize_L2(search_vectors)

    k = index.ntotal
    D, I = index.search(search_vectors, k=k)  # return Distance, Index
    # save the result
    faiss.write_index(index, "./example_db/doc_index.faiss")
    """
    def __init__(self, embedding_fn, db_uri: str, normalize: bool = False):
        self.embedding_fn = embedding_fn
        self.emb_dim = embedding_fn.embedding_dim
        self.db_path = db_uri
        self.normalize = normalize
        self.index = None
        self.index_to_docid = dict()
        self.docid_to_doc = dict()
        self.doc_fname_to_docid = dict()

        # init db
        self.init_db()
        self.init_db_size = self.index.ntotal
        self.existed_doc_ids = set(self.docid_to_doc.keys())

    def init_db(self):
        self.index = faiss.IndexFlatL2(self.emb_dim)
        if os.path.exists(self.db_path):
            index, docid_to_doc, index_to_docid, doc_fname_to_docid = self.load_local(self.db_path)
            self.index.merge_from(index)
            self.docid_to_doc.update(docid_to_doc)
            self.index_to_docid.update(index_to_docid)
            self.doc_fname_to_docid.update(doc_fname_to_docid)

    def _set_doc_index_mapping(self, texts: List[str]):
        doc_ids = self._gen_uuid(len(texts), exist_ids=self.existed_doc_ids)
        docid_to_doc = {i: text for i, text in zip(doc_ids, texts)}  # {doc_id: doc_text}
        index_to_docid = {(i + self.init_db_size): doc_id for i, doc_id in enumerate(doc_ids)}  # {index: doc_id}

        # update to exist mapping
        self.docid_to_doc.update(docid_to_doc)
        self.index_to_docid.update(index_to_docid)
        self.existed_doc_ids.update(set(doc_ids))
        return docid_to_doc, index_to_docid

    def create(self, texts: List[str], doc_fname: str = "default"):
        # create search vectors
        vectors = self.embedding_fn.embedding(texts)
        vectors = vectors.astype(np.float32)  # TypeError: in method 'fvec_renorm_L2', argument 3 of type 'float *'
        # if normalize
        if self.normalize:
            faiss.normalize_L2(vectors)
        self.index.add(vectors)

        # doc / index id mapping
        docid_to_doc, index_to_docid = self._set_doc_index_mapping(texts)
        # store doc_fname
        if doc_fname not in self.doc_fname_to_docid:
            self.doc_fname_to_docid.update({doc_fname: list(docid_to_doc.keys())})
        elif doc_fname == "default":
            self.doc_fname_to_docid[doc_fname] = self.doc_fname_to_docid[doc_fname].extend(list(docid_to_doc.keys()))
        return {'update': len(texts), 'docid_to_doc': docid_to_doc, 'index_to_docid': index_to_docid}

    def similarity_search_with_score(self, search_text: str, k: Optional[int] = None) -> List[Dict]:
        r"""
        Search a single question

        Return:
            - [{id, distance, text}]
        """
        search_vectors = self.embedding_fn.embedding([search_text])
        search_vectors = search_vectors.astype(np.float32)
        faiss.normalize_L2(search_vectors)
        k = self.index.ntotal if k is None else k  # k means find nearest k documents
        D, I = self.index.search(search_vectors, k=k)  # return Distance, Index; example [[D1, D2]], [[I1, I2]]

        search_res = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            _text = self.docid_to_doc[self.index_to_docid[idx]]
            _res = {'id': idx, 'distance': dist, 'text': _text}
            search_res.append(_res)
        return search_res

    def check_by_doc_fname(self, doc_fname: str):
        return doc_fname in self.doc_fname_to_docid

    def save_local(self, db_path: str):
        r"""
        db_path example: "faiss_db.faiss"
        """
        # save index, docstore and index to doc_ids (pickle)
        with open(db_path, "wb") as f:
            data = (self.index, self.docid_to_doc, self.index_to_docid, self.doc_fname_to_docid)
            dill.dump(data, f)

    def load_local(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(r"file {db_path} not found.")

        with open(db_path, "rb") as f:
            index, docid_to_doc, index_to_docid, doc_fname_to_docid = dill.load(f)
        return index, docid_to_doc, index_to_docid, doc_fname_to_docid   

    def delete(self, ids: Union[List, np.ndarray]) -> List[Dict]:
        remove_docs = []
        for i in ids:
            pop_docid = self.index_to_docid.pop(i)
            pop_doc = self.docid_to_doc.pop(pop_docid)

            _doc = {'id': i, 'text': pop_doc}
            remove_docs.append(_doc)

        # remove
        self.index.remove_ids(np.array(ids))
        return remove_docs

    def _gen_uuid(self, n: int, exist_ids: Optional[Set] = None):
        if len(exist_ids) == 0 or exist_ids is None:
            return [str(uuid.uuid4()) for _ in range(n)]

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
