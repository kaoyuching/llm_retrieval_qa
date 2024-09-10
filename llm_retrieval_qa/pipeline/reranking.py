from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class ReRanking():
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def maxsim(self, query_emb, doc_emb):
        query_emb = query_emb.unsqueeze(2)  # (bs, q_len, emb_size) -> (bs, q_len, 1, emb_size)
        doc_emb = doc_emb.unsqueeze(1)  # (bs, doc_len, emb_size) -> (bs, 1, doc_len, emb_size)

        # cosine similarity
        sim_matrix = F.cosine_similarity(query_emb, doc_emb, dim=-1)

        max_sim_scores, _ = torch.max(sim_matrix, dim=2)

        arg_max_sim = torch.mean(max_sim_scores, dim=1)
        return arg_max_sim

    def rank(self, query: List[str], documents: List[str]):
        res = []
        query_encoding = self.tokenizer(query, return_tensors="pt")
        # query_emb = self.model(**query_encoding).last_hidden_state.mean(dim=1)
        query_emb = self.model(**query_encoding).last_hidden_state.squeeze(0)

        for doc in documents:
            doc_encoding = self.tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            doc_emb = self.model(**doc_encoding).last_hidden_state

            score = self.maxsim(query_emb.unsqueeze(0), doc_emb)
            _res = {"score": score.item(), "text": doc}
            res.append(_res)
        res = sorted(res, key=lambda x: x["score"], reverse=True)
        return res
