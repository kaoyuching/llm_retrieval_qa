import os
import json
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import WebSocket

from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from llm_retrieval_qa.configs import settings, model_config, vector_store_config,\
    get_prompt_template, get_embedding_fn
from llm_retrieval_qa.splitter import split_html
from llm_retrieval_qa.pipeline.model_loader import load_model
from api.routers import router


app = FastAPI()
app.include_router(router)

app.mount("/static", StaticFiles(directory="api/static"), name="static")


# embedding
emb_cfg = model_config["embedding_cfgs"]
embedding_fn = get_embedding_fn(emb_cfg)

# prepare prompt template
prompt_template_fn, full_prompt_template = get_prompt_template(model_config["prompt_template"])

# load model
model = load_model(model_config, settings.quantization, settings.device)
top_k = settings.search_topk


# load vector db
def load_vector_db():
    if vector_store_config.type == "milvus":
        from llm_retrieval_qa.vector_store.milvus import DbMilvus

        vector_db = DbMilvus(
            embedding_fn,
            vector_store_config.uri,
            db_name=vector_store_config.db_name,
            collection_name=vector_store_config.collection,
        )
    elif vector_store_config.type == "faiss":
        from llm_retrieval_qa.vector_store.faiss import DbFAISS
        
        vector_db = DbFAISS(embedding_fn, vector_store_config.uri, normalize=True)
    return vector_db


def get_streaming_fn():
    vector_db = load_vector_db()

    if model_config["format"] == "hf":
        from transformers import AutoTokenizer
        from llm_retrieval_qa.pipeline.streaming import QAHFStreamer, generate_response

        tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"])

        streamer_config = {
            'args': (tokenizer,),
            'kwargs': model_config["streamer"]
        }

        llm_model_runtime_kwargs = model_config["runtime"]
        if 'return_full_text' in llm_model_runtime_kwargs:
            llm_model_runtime_kwargs.pop('return_full_text')

        model_kwargs = dict(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            **llm_model_runtime_kwargs,
        )

        qa_streaming = QAHFStreamer(
            tokenizer,
            model,
            streamer_config,
            vector_db,
            prompt_template_fn,
            top_k=top_k,
            return_source_documents=False,
            similarity_score_threshold=None,
            model_kwargs=model_kwargs,
            device=settings.device,
        )
    elif model_config["format"] == "gguf":
        from llm_retrieval_qa.pipeline.streaming import LlamaCppStreamer, generate_response

        qa_streaming = LlamaCppStreamer(
            model,
            vector_db,
            prompt_template_fn,
            streamer_cfg=model_config["streamer"],
            top_k=top_k,
            return_source_documents=False,
            similarity_score_threshold=None,
            model_kwargs=model_config["generate"],
        )
    return qa_streaming


async def generator(question: str, send_json: bool = False):
    qa_streaming = get_streaming_fn()
    qa_streaming(question)
    for x in qa_streaming.streamer:
        data = {'text': x}
        if send_json:
            yield data
        else:
            yield f"event: message\ndata: {json.dumps(data)}\n\n"


@app.get('/answer_stream')
async def stream(question: str = Query()):
    return StreamingResponse(generator(question, send_json=False), media_type="text/event-stream")


@app.websocket("/answer_stream_ws")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    question = await websocket.receive_text()
    async for text in generator(question, send_json=True):
        try:
            await websocket.send_json(text)
        except (WebSocketDisconnect, ConnectionClosedOK):
            print('ws Disconnected!', flush=True)
            break
    await websocket.close()
