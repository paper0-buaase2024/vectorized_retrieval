import torch
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import requests

# 关闭安全请求警告
requests.packages.urllib3.disable_warnings()

username = 'elastic'
password = 'NKp1ZqZS-oOMJ+4I_nPL'           # ES密码
host = 'https://114.116.194.36:9200'        # ES HOST
index = 'papers_vector'

# 加载模型
model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
if torch.cuda.is_available():
    # 使用GPU推理
    model = model.to(torch.device("cuda"))


# 使用IK分词器
def ik_search(es, index_name, query_text):
    source_fields = ["text_field"]
    query = {
        "match": {
            "text_field": query_text
        }
    }

    resp = es.search(
        index=index_name,
        fields=source_fields,
        query=query,
        size=10,
        source=False)
    return resp


# 使用k-NN搜索
def knn_search(es, index_name, vector):
    source_fields = ["text_field"]
    knn = [{
        "field": "vector",
        "query_vector": vector,
        "k": 10,
        "num_candidates": 100
    }]

    resp = es.search(
        index=index_name,
        fields=source_fields,
        knn=knn,
        source=False)
    return resp


# 生成句子向量
def gen_vector(sentences):
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return embeddings


# 连接ES
es = Elasticsearch(hosts=[host], basic_auth=(username, password), verify_certs=False)

query = "arabic recognition"

# 调用ik检索
ik_resp = ik_search(es, index, query)

# 调用knn检索
knn_resp = knn_search(es, index, gen_vector(query))

counter = 1
for hit in ik_resp['hits']['hits']:
    text_field = hit['fields']['text_field'][0]
    print(f"{counter}. {text_field}")
    counter += 1

counter = 1
for hit in knn_resp['hits']['hits']:
    text_field = hit['fields']['text_field'][0]
    print(f"{counter}. {text_field}")
    counter += 1

