from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from datetime import datetime
import torch
import json
import requests
import re
# 关闭安全请求警告
requests.packages.urllib3.disable_warnings()

# elastic settings
es_username = 'elastic'
es_password = 'NKp1ZqZS-oOMJ+4I_nPL'        # ES密码
es_host = '114.116.194.36'                  # ES HOST
es_port = 9200
es = Elasticsearch(
    hosts=[{'host': es_host, 'port': es_port, 'scheme': 'http'}],
    basic_auth=(es_username, es_password),
    verify_certs=False,
    request_timeout=60
)

# index_name = 'test_vector'                  # 测试用
index_name = 'papers_vector'              # 正式用

# 加载模型
model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')  # 后续自己训练一个？
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))


# 生成向量
def gen_vector(sentences):
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return embeddings


# 读取file, 用于批量插入
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


# ---------- Insert Related ----------- #
# for insert
# id, authors, title, abstract: string
# date: yyyy-mm-dd
# 测试通过: 2023-04-27
def papers_insert(doc_id, date, authors, title, abstract):
    es.index(index=index_name,
             id=doc_id,
             document={
                'id': doc_id,
                'date': date,
                'authors': authors,
                'title': title,
                'abstract': abstract,
                'text_field': authors + '\n' + title + '\n' + abstract,
                'vector': gen_vector(authors + '\n' + title + '\n' + abstract)
                })


# for bulk insert
# 测试通过: 2023-04-27
def papers_bulk_insert(file_path, chunk_size=1000):
    data = read_data(file_path)
    bulk(es, data, chunk_size=chunk_size)


# ---------- Delete Related ----------- #
# for delete
# id: string
# 测试通过: 2023-04-27
def papers_del(doc_id):
    es.delete(index=index_name,
              id=doc_id)


# for bulk delete
# doc_id_list: list
# 测试通过: 2023-04-27
def papers_bulk_del(doc_id_list):
    for doc_id in doc_id_list:
        es.delete(index=index_name,
                  id=doc_id)


# for clear (清空)
# 测试通过: 2023-04-27
def papers_clear():
    query = {
        "query": {
            "match_all": {}
        }
    }
    es.delete_by_query(index=index_name, body=query)


# ---------- Search Related ----------- #
# for vectorized_retrieval (向量化检索)
# id: string
# 测试通过: 2023-04-27
def papers_knn_search(query_text, date_from=None, date_to=None):
    source_fields = ['title']

    knn_filter = []

    if date_from is not None:
        knn_filter.append({
            "range": {
                "date": {
                    "gte": datetime.strptime(date_from, "%Y-%m-%d")
                }
            }
        })

    if date_to is not None:
        knn_filter.append({
            "range": {
                "date": {
                    "lte": datetime.strptime(date_to, "%Y-%m-%d")
                }
            }
        })

    knn = [{
        "field": "vector",
        "query_vector": gen_vector(query_text),
        "k": 50,
        "num_candidates": 100,
        "similarity": 0.30,
        "filter": knn_filter
    }]

    resp = es.search(
        index=index_name,
        fields=source_fields,
        knn=knn,
        size=50,
        source=False)
    return resp


# for IK retrieval (分词检索)
# id: string
# 测试通过: 2023-04-27
def papers_ik_search(query_text, date_from=None, date_to=None):
    source_fields = ['title']

    ik_filter = []

    if date_from is not None:
        ik_filter.append({
            "range": {
                "date": {
                    "gte": datetime.strptime(date_from, "%Y-%m-%d")
                }
            }
        })

    if date_to is not None:
        ik_filter.append({
            "range": {
                "date": {
                    "lte": datetime.strptime(date_to, "%Y-%m-%d")
                }
            }
        })

    query = {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["id^2", "text_field"],
                    "boost": 1
                }
            },
            "filter": ik_filter
        }
    }

    resp = es.search(
        index=index_name,
        fields=source_fields,
        query=query,
        size=50,
        source=False)
    return resp


def is_arxiv_id(string):
    arxiv_pattern = r'^\d{4}\.\d{4,5}$'
    match = re.match(arxiv_pattern, string)
    return bool(match)


def papers_mix_search(query_text, date_from='', date_to='', k=50):
    source_fields = ['id']
    mix_filter = []
    if date_from != '':
        mix_filter.append({
            "range": {
                "date": {
                    "gte": datetime.strptime(date_from, "%Y-%m-%d")
                }
            }
        })
    if date_to != '':
        mix_filter.append({
            "range": {
                "date": {
                    "lte": datetime.strptime(date_to, "%Y-%m-%d")
                }
            }
        })

    if is_arxiv_id(query_text):
        query = {
            "bool": {
                "must": [
                    {
                        "match": {
                            "id": query_text
                        }
                    }
                ],
                "filter": mix_filter
            }
        }
        resp = es.search(
            index=index_name,
            fields=source_fields,
            query=query,
            source=False)
        return resp

    query = {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["id", "title^4", "authors^2", "text_field"],
                    "boost": 1
                }
            },
            "filter": mix_filter
        }
    }
    knn = [{
        "field": "vector",
        "query_vector": gen_vector(query_text),
        "k": k,
        "num_candidates": k*2,
        "similarity": 0.30,
        "boost": 32,
        "filter": mix_filter
    }]
    resp = es.search(
        index=index_name,
        fields=source_fields,
        query=query,
        knn=knn,
        size=k,
        min_score=15,
        source=False)
    return resp


resp = papers_mix_search('2402.01771', date_from='', date_to='')
print(len(resp['hits']['hits']))
print(resp)
