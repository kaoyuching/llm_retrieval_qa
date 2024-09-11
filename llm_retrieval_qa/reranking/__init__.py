

def get_reranking_class(class_name):
    if class_name == "ONNXReranking":
        from . import onnx as m
    elif class_name == "HFReranking":
        from . import huggingface as m
    else:
        return ValueError(...)
    return getattr(m, class_name)
