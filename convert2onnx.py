import os
import torch
import onnx
from onnxconverter_common import float16
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import BertTokenizer, BertModel


r"""
Reference:
- convert llama2 to onnx: https://github.com/tpoisonooo/llama.onnx/blob/main/tools/export-onnx.py
"""


# model_name = "GanymedeNil/text2vec-large-chinese"
model_name = "colbert-ir/colbertv2.0"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

onnx_model_outpath = "../models/colbertv2.0/test_model.onnx"
if not os.path.exists(onnx_model_outpath):
    dirname = os.path.dirname(onnx_model_outpath)
    os.makedirs(dirname, exist_ok=True)
# sample_input = (torch.ones(2, 512, dtype=torch.int64), torch.randn(2, 512), torch.ones(2, 512, dtype=torch.int64))
sample_input = (torch.ones(2, 512, dtype=torch.int64), torch.randn(2, 512))

torch.onnx.export(
    model.eval(),
    sample_input,
    onnx_model_outpath,
    export_params=True,
    do_constant_folding=True,
    opset_version=12,
    # input_names=["input_ids", "attention_mask", "token_type_ids"],
    input_names=["input_ids", "attention_mask"],
    # output_names=["logits"],
    output_names=["contextual"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence_length"},
        "attention_mask": {0: "batch", 1: "sequence_length"},
        # "token_type_ids": {0: "batch", 1: "sequence_length"},
    },
    verbose=True,
)
