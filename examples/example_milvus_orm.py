import sys
sys.path.append("./")
from pymilvus import MilvusClient
from pymilvus import connections, db, utility
from pymilvus import DataType, FieldSchema, CollectionSchema
from pymilvus import Collection

from llm_retrieval_qa.vector_store import HFEmbedding


r"""
Use Milvus ORM
"""


model_name = "GanymedeNil/text2vec-large-chinese"
hf_embedding = HFEmbedding(model_name, normalize_embeddings=True)

conn_name = "demo"
conn = connections.connect(alias=conn_name, host="127.0.0.1", port="19530")
print(connections.list_connections())

db_name = "docs_db"
if db_name not in db.list_database(using=conn_name):
    database = db.create_database(db_name, using=conn_name)
db.using_database(db_name, using=conn_name)
print(connections.list_connections())


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
collection_schema = CollectionSchema(fields, auto_id=False, enable_dynamic_field=True)

# manage collections
collection_name = "data_collection"

# check if collection exists
exist_collection = utility.has_collection(collection_name, using=conn_name)
if not exist_collection:
    collection = Collection(collection_name, schema=collection_schema, using=conn_name, num_shards=2)

    # create index on target field
    collection.create_index(
        field_name="vector",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        },
        index_name="vector_index",
    )

# load collection
collection.load()

# insert data into the given collection
texts = ["How are you?", "The weather is good today."]
vectors = hf_embedding.embedding(texts)  # shape (2, 1024)
data = [{"vector": vector, "text": text} for vector, text in zip(vectors, texts)]
insert_res = collection.insert(data=data)
print(insert_res)

# query data
query_res = collection.query(expr="id in [1]", output_fields=field_names)
print(query_res)

# release collection
collection.release()
