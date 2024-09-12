from pymilvus import MilvusClient


db_name = "docs_db"

client = MilvusClient(uri="http://localhost:19530")
client.using_database(db_name)
print(client.list_collections())

collection_name = "data_collection"
client.drop_collection(collection_name)
print(client.list_collections())
