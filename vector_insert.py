from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
import torch
import json
import requests

# 关闭安全请求警告
requests.packages.urllib3.disable_warnings()

es_username = 'elastic'
es_password = 'NKp1ZqZS-oOMJ+4I_nPL'        # ES密码
es_host = '114.116.194.36'                  # ES HOST
es_port = 9200

# 加载模型
model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')  # 后续自己训练一个？
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))

es = Elasticsearch(
    hosts=[{'host': es_host, 'port': es_port, 'scheme': 'https'}],
    basic_auth=(es_username, es_password),
    verify_certs=False,
    request_timeout=6000
)

# 读取文本
file_path = 'cscl.json'
index_name = 'papers_vector'


def gen_vector(sentences):
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return embeddings


def read_data(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            try:
                id = data.get('id', '')
                date = data.get('update_date', '')
                authors = data.get('authors', '')
                title = data.get('title', '')
                abstract = data.get('abstract', '')
                yield {
                    '_index': index_name,
                    '_id': id,
                    '_source': {
                        'id': id,
                        'date': date,
                        'authors': authors,
                        'title': title,
                        'abstract': abstract,
                        'text_field': authors + '\n' + title + '\n' + abstract,
                        'vector': gen_vector(authors + '\n' + title + '\n' + abstract)
                    }
                }
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)


# 执行批量插入
def bulk_insert(file_path, chunk_size=1000):
    data = read_data(file_path)
    bulk(es, data, chunk_size=chunk_size)


bulk_insert(file_path)
