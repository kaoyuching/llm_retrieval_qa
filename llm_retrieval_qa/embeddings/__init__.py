# from .onnx import OnnxEmbedding
# from .huggingface import HFEmbedding

def get_embedding_class(class_name):
    if class_name == "OnnxEmbedding":
        from . import onnx as m
    elif class_name == "HFEmbedding":
        from . import huggingface as m
    else:
        return ValueError(...)
    return getattr(m, class_name)
