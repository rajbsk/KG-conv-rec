from collections import defaultdict
import pickle
import dgl
import torch

def load_pickle_file(location):
    with open(location, "rb") as f:
        pickle_variable = pickle.load(f)
    return pickle_variable

def _read_knowledge_graph(kg_file, entity2entityId, relation2relationId):
    triples = set()

    for line in open(kg_file, "r"):
        triples.add(line[:-1])
        line = line[:-1].split("\t")

    heads = []
    tails = []
    relations = []
    for line in triples:
        line = line.split("\t")
        heads.append(entity2entityId[line[0]])
        tails.append(entity2entityId[line[2]])
        relations.append(relation2relationId[line[1]])
        
    return (heads, tails, relations)

def _get_triples(kg):
    triples = []
    for entity in kg:
        for relation, tail in kg[entity]:
            if entity != tail:
                triples.append([entity, relation, tail])
    return triples

def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    entityIds = [v for k, v in entity2entityId.items()]
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = int(line[0])
            if entity not in entityIds:
                continue
            entityId = entity
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

def _edge_list(kg):
    edge_list = []
    for entity in kg.keys():
        for tail_and_relation in kg[entity]:
            if entity != tail_and_relation[1]:
                edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 10 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 10], len(relation_idx)

def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    entityIds = [v for k, v in entity2entityId.items()]
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = int(line[0])
            if entity not in entityIds:
                continue
            entityId = entity
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings