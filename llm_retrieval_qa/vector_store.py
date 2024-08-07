from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import faiss
from pymilvus import MilvusClient
from pymilvus import connections, db


class HFEmbedding():
    def __init__(self, model_path: str, normalize_embeddings: bool = True, device_map: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, device_map=device_map)
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = self.model.get_input_embeddings().embedding_dim

    def encode(self, sentences: List[str]) -> List[float]:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        return encoded_input

    def embedding(self, sentences: List[str]):
        encoded_input = self.encode(sentences)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embedding_output = self.mean_pooling(model_output, encoded_input['attention_mask'])
        if self.normalize_embeddings:
            embedding_output = F.normalize(embedding_output, p=2, dim=1)
        return embedding_output

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
