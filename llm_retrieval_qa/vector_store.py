from typing import List, Optional
import atexit
import json
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import faiss
from pymilvus import MilvusClient
from pymilvus import db
from pymilvus import DataType, FieldSchema, CollectionSchema

from tokenizers import Tokenizer
import onnx
import onnxruntime


class HFEmbedding():
    def __init__(self, model_path: str, normalize_embeddings: bool = True, device_map: str = "cpu"):
        self.device = device_map
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, device_map=self.device)
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = self.model.get_input_embeddings().embedding_dim
        self.pos_emb_dim = self.model.config.max_position_embeddings

    def encode(self, sentences: List[str]) -> List[float]:
        sentences = list(map(lambda x: x.replace("\n", " "), sentences))
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.pos_emb_dim,
            return_tensors='pt'
        )
        return encoded_input

    def embedding(self, sentences: List[str]):
        encoded_input = self.encode(sentences).to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embedding_output = self.mean_pooling(model_output, encoded_input['attention_mask'])
        if self.normalize_embeddings:
            embedding_output = F.normalize(embedding_output, p=2, dim=1)
        return embedding_output.cpu().numpy()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class OnnxEmbedding():
    def __init__(self, model_path: str, tokenizer_path: str, tokenizer_cfg_path: str, normalize_embeddings: bool = True, device_map: str = "cpu"):
        self.device = device_map
        with open(tokenizer_cfg_path, "r") as f:
            token_cfg = json.load(f)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(
            token_cfg["max_length"],
            strategy=token_cfg.get("truncation_strategy", "longest_first"),
            direction=token_cfg.get("truncation_side", "right"),
        )

        sess_options = onnxruntime.SessionOptions()
        providers = ['CPUExecutionProvider']
        self.model = onnxruntime.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.normalize_embeddings = normalize_embeddings

        # get embedding dimentions
        self.embedding_dim = None
        self.pos_emb_dim = None

        onnx_model = onnx.load(model_path)
        initializers = onnx_model.graph.initializer
        for _init in initializers:
            if 'word_embeddings' in _init.name:
                self.embedding_dim = _init.dims[-1]
            if 'position_embeddings' in _init.name:
                self.pos_emb_dim = _init.dims[0]

    def encode(self, sentences: List[str]) -> List[float]:
        sentences = list(map(lambda x: x.replace("\n", " "), sentences))
        encoded_input = self.tokenizer.encode_batch(sentences)
        return encoded_input

    def encode_post_process(self, encoded_res: List):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for res in encoded_res:
            input_ids.append(res.ids)
            attention_mask.append(res.attention_mask)
            token_type_ids.append(res.type_ids)
        input_ids = np.stack(input_ids, axis=0)
        attention_mask = np.stack(attention_mask, axis=0)
        token_type_ids = np.stack(token_type_ids, axis=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def embedding(self, sentences: List[str]):
        encoded_input = self.encode(sentences)
        encoded_input = self.encode_post_process(encoded_input)

        model_output = self.model.run(None, encoded_input)
        embedding_output = self.mean_pooling(model_output, encoded_input['attention_mask'])
        if self.normalize_embeddings:
            embedding_output = self.normalize(embedding_output)
        return embedding_output

    def mean_pooling(self, model_output, attention_mask):
        r"""
        emb: shape (batch, num_tokens, emb_dim)
        attn_mask: shape (batch. num_tokens)
        """
        token_embeddings = model_output[0]
        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, 2), token_embeddings.shape)
        input_mask_sum = np.sum(input_mask_expanded, axis=1)
        return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(input_mask_sum, 1e-9, None)

    def normalize(self, x, eps: float = 1e-9):
        norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        return x / np.clip(norm, eps, None)


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
        fields = [id_field, vector_field, text_field, doc_fname_field]
        self.field_names = ["id", "vector", "text", "doc_fname"]
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
        data = [{"vector": vector, "text": text, "doc_fname": doc_fname} for vector, text in zip(vectors, texts)]
        res = self.client.insert(self.collection_name, data)
        return res

    def similarity_search_with_score(self, data: str, top_k: int = 10, **kwargs):
        r"""
        search single question

        Return:
            - id, distance, entity
        """
        vector = self.embedding_fn.embedding([data])  # shape: (batch, embedding size)
        res = self.client.search(
            self.collection_name,
            data=vector,
            limit=top_k,
            output_fields=["text", "doc_fname"],
            search_params={"metric_type": "COSINE"}
        )
        return res[0]

    def get(self, filter: str = ""):
        res = self.client.query(
            self.collection_name,
            filter=filter,
            output_fields=self.field_names,
        )
        return res
