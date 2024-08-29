import sys
sys.path.append("../")
from pymilvus import MilvusClient
from pymilvus import connections, db
from pymilvus import DataType, FieldSchema, CollectionSchema

from llm_retrieval_qa.vector_store import HFEmbedding


r"""
A Milvus cluster supports a maximum of 64 databases.
collections are created in the given database.
schema is used to define the properties of a collection and the fields within.
"""


model_name = "GanymedeNil/text2vec-large-chinese"
hf_embedding = HFEmbedding(model_name, normalize_embeddings=True)

client = MilvusClient(uri="http://localhost:19530")
conn_name = client._using

db_name = "docs_db"
if db_name not in db.list_database(using=conn_name):
    client.create_database(db_name)
client.using_database(db_name)
print("existing collections:", client.list_collections())

r"""
Milvus supports only one primary key field in a collection.
"""
# define field schema
emb_dim = hf_embedding.embedding_dim
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="primary id")
vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=emb_dim, description="embedding vector")
text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, description="document content")
fields = [id_field, vector_field, text_field]
field_names = ["id", "vector", "text"]

# define collection schema
collection_schema = CollectionSchema(fields, auto_id=True, enable_dynamic_field=True)

# manage collections
collection_name = "data_collection"

# check if collection exists
exist_collection = client.has_collection(collection_name)
if not exist_collection:
    client.create_collection(collection_name, schema=collection_schema)

    # create index on target field
    client.create_index(
        collection_name,
        index_params=[{
            "field_name": "vector",
            "index_name": "vector_index",
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }]
    )

# load collection
client.load_collection(collection_name)

# insert data into the given collection
texts = ["How are you?", "The weather is good today."]
vectors = hf_embedding.embedding(texts)  # shape (2, 1024)
data = [{"vector": vector, "text": text} for vector, text in zip(vectors, texts)]
insert_res = client.insert(collection_name, data=data)

# query data with client
query_res = client.query(collection_name, filter="", output_fields=["id"])

get_res = client.get(collection_name, ids=[0])
print("get data:", get_res)

# release collection
client.release_collection(collection_name)

# drop collection
client.drop_collection(collection_name)
