from typing import List
import re
import json
import numpy as np

from tokenizers import Tokenizer
import onnx
from onnx.mapping import TENSOR_TYPE_MAP
import onnxruntime


class ONNXReranking():
    _tensor_type_mapping = {v.name.split('.')[-1].lower(): v.np_dtype for k, v in TENSOR_TYPE_MAP.items()}

    def __init__(self, model_path: str, tokenizer_path: str, tokenizer_cfg_path: str):
        with open(tokenizer_cfg_path, "r") as f:
            token_cfg = json.load(f)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(
            token_cfg.get("model_max_length", 512),
        )

        sess_options = onnxruntime.SessionOptions()
        providers = ['CPUExecutionProvider']
        self.model = onnxruntime.InferenceSession(model_path, sess_options=sess_options, providers=providers)

    def _tensor_dtype_to_np_dtype(self, type_name: str):
        r"""
        'tensor(float)' -> np.float64
        """
        str_type = re.search('(?<=tensor\()\w*', type_name).group(0)
        return self._tensor_type_mapping[str_type]

    def cosine_similarity(self, x1, x2, dim: int = 1, eps: float = 1e-8):
        _mag_x1 = np.sum(x1 ** 2, axis=dim, keepdims=True)
        _mag_x2 = np.sum(x2 ** 2, axis=dim, keepdims=True)
        
        mag_x1 = np.sqrt(np.clip(_mag_x1, eps, None))
        mag_x2 = np.sqrt(np.clip(_mag_x2, eps, None))

        norm_x1 = x1 / mag_x1
        norm_x2 = x2 / mag_x2
        return np.sum(norm_x1 * norm_x2, axis=dim)

    def maxsim(self, query_emb: np.ndarray, doc_emb: np.ndarray):
        query_emb = np.expand_dims(query_emb, axis=2)  # (bs, q_len, emb_size) -> (bs, q_len, 1, emb_size)
        doc_emb = np.expand_dims(doc_emb, axis=1)  # (bs, doc_len, emb_size) -> (bs, 1, doc_len, emb_size)

        # cosine similarity
        sim_matrix = self.cosine_similarity(query_emb, doc_emb, dim=-1)  # (bs, q_len, doc_len)
        max_sim_scores = np.max(sim_matrix, axis=2)
        arg_max_sim = np.mean(max_sim_scores, axis=1)
        return arg_max_sim

    def encode_post_process(self, encoded_res: List):
        r"""
        convert tokens to onnx input
        """
        input_ids = []
        attention_mask = []
        for res in encoded_res:
            input_ids.append(res.ids)
            attention_mask.append(res.attention_mask)
        input_ids = np.stack(input_ids, axis=0)
        attention_mask = np.stack(attention_mask, axis=0)
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        onnx_input_data = {
            node.name: data[node.name].astype(self._tensor_dtype_to_np_dtype(node.type))
            for node in self.model.get_inputs()
        }
        return onnx_input_data

    def rank(self, query: List[str], documents: List[str]):
        res = []
        query_encoding = self.tokenizer.encode_batch(query)
        query_encoding = self.encode_post_process(query_encoding)

        query_emb = self.model.run(None, query_encoding)[0]

        for doc in documents:
            doc_encoding = self.tokenizer.encode_batch([doc])
            doc_encoding = self.encode_post_process(doc_encoding)
            doc_emb = self.model.run(None, doc_encoding)[0]

            score = self.maxsim(query_emb, doc_emb)
            _res = {"score": score.item(), "text": doc}
            res.append(_res)
        res = sorted(res, key=lambda x: x["score"], reverse=True)
        return res
