from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


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
