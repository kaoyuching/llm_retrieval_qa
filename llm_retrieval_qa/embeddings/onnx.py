from typing import List
import json
import numpy as np

from tokenizers import Tokenizer
import onnx
import onnxruntime


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
