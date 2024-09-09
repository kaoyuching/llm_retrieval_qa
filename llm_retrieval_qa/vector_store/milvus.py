from typing import List, Optional, Dict
import atexit

from pymilvus import MilvusClient
from pymilvus import db
from pymilvus import DataType, FieldSchema, CollectionSchema


class DbMilvus():
    def __init__(
        self,
        embedding_fn,
        db_uri: str,
        db_name: str = "default",
        collection_name: str = "default",
        normalize: bool = False,
    ):
        self.embedding_fn = embedding_fn
        self.emb_dim = embedding_fn.embedding_dim
        self.normalize = normalize
        self.client = MilvusClient(uri=db_uri)
        self.conn_name = self.client._using
        self.db_name = db_name
        self.collection_name = collection_name
        self.field_names = None
        self.init_db(self.db_name)
        self.create_collection(collection_name)
        self.client.load_collection(collection_name)

        @atexit.register
        def release():
            self.client.release_collection(collection_name)

    def init_db(self, db_name: str = "default"):
        if self.db_name not in db.list_database(using=self.conn_name):
            database = self.client.create_database(db_name)
        self.client.using_database(db_name)

    def _collection_schema(self):
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="primary id")
        vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.emb_dim, description="embedding vector")
        text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, description="document content")
        doc_fname_field = FieldSchema(name="doc_fname", dtype=DataType.VARCHAR, max_length=65535, description="document file name")
        doc_id_field = FieldSchema(name="doc_id", dtype=DataType.INT64, description="document id")
        fields = [id_field, vector_field, text_field, doc_fname_field]
        self.field_names = ["id", "vector", "text", "doc_fname", "doc_id"]
        collection_schema = CollectionSchema(fields, auto_id=True, enable_dynamic_field=True)
        return collection_schema

    def create_collection(self, collection_name: str):
        exist_collection = self.client.has_collection(collection_name)
        if not exist_collection:
            self.client.create_collection(collection_name, schema=self._collection_schema())

            self.client.create_index(
                collection_name,
                index_params=[{
                    "field_name": "vector",
                    "index_name": "vector_index",
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                }]
            )

    def create(self, texts: List, doc_fname: str = "default"):
        r"""
        return:
            Dict: Number of rows that were inserted and the inserted primary key list.
        """
        vectors = self.embedding_fn.embedding(texts)  # shape (n, emb_dim)
        data = [{"vector": vector, "text": text, "doc_fname": doc_fname, "doc_id": i} for i, (vector, text) in enumerate(zip(vectors, texts))]
        res = self.client.insert(self.collection_name, data)
        return res

    def similarity_search_with_score(self, data: str, top_k: int = 10, **kwargs) -> List[Dict]:
        r"""
        search single question

        Return:
            - id, distance, entity[text, doc_fname, doc_id]
        """
        vector = self.embedding_fn.embedding([data])  # shape: (batch, embedding size)
        res = self.client.search(
            self.collection_name,
            data=vector,
            limit=top_k,
            output_fields=["text", "doc_fname", "doc_id"],
            search_params={"metric_type": "COSINE"}
        )  # pymilvus.client.types.ExtraList[List[Dict]]
        return res[0]

    def get(self, filter: str = "", limit_n: Optional[int] = None) -> List[Dict]:
        if limit_n is None:
            limit_n = max(self.ntotal, 1)

        res = self.client.query(
            self.collection_name,
            filter=filter,
            output_fields=self.field_names,
            limit=limit_n,
        )  # pymilvus.client.types.ExtraList[Dict]
        return res

    def get_by_id(self, idx: int) -> Dict:
        res = self.client.get(self.collection_name, ids=[idx])[0]  # pymilvus.client.types.ExtraList[Dict]
        emb = res.get('vector', [])
        text = res.get('text', '')
        return {'id': idx, 'text': text, 'embedding': emb}

    @property
    def ntotal(self):
        return self.client.get_collection_stats(self.collection_name)['row_count']

    def doc_ntotal(self, doc_fname: str):
        res = self.get(filter=f"doc_fname=='{doc_fname}'")
        return len(res)
