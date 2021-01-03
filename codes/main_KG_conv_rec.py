from __future__ import print_function, division
import sys

import os
import torch
from skimage import io, transform
import numpy as np
import pickle
import glob
from tqdm import tqdm

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataset import KGCRDataset, ToTensor, pad_collate_entities
from trainer import KGCRTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data_directory = "../dataset/data_2/redial/"
    opt_dataset_train = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "id2entity": data_directory+"id2entity.pkl", "text_dict": data_directory+"text_dict.pkl", "kg": data_directory+"subkg.pkl",
                    "dataset": data_directory+"dataset_train.pkl", "knowledge_graph": data_directory+"opendialkg_triples.txt",
                    "n_hop": 2, "movie_ids": data_directory+"movie_ids.pkl"}
    opt_dataset_dev = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "id2entity": data_directory+"id2entity.pkl", "text_dict": data_directory+"text_dict.pkl", "kg": data_directory+"subkg.pkl",
                    "dataset": data_directory+"dataset_valid.pkl", "knowledge_graph": data_directory+"opendialkg_triples.txt",
                    "n_hop": 2, "movie_ids": data_directory+"movie_ids.pkl"}
    opt_dataset_test = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                    "id2entity": data_directory+"id2entity.pkl", "text_dict": data_directory+"text_dict.pkl", "kg": data_directory+"subkg.pkl",
                    "dataset": data_directory+"dataset_test.pkl", "knowledge_graph": data_directory+"opendialkg_triples.txt",
                    "n_hop": 2, "movie_ids": data_directory+"movie_ids.pkl"}

    # Dataset Preparation
    KGCR_dataset_train = KGCRDataset(opt=opt_dataset_train, transform=transforms.Compose([ToTensor(opt_dataset_train)]))
    KGCR_dataset_dev = KGCRDataset(opt=opt_dataset_dev, transform=transforms.Compose([ToTensor(opt_dataset_dev)]))
    KGCR_dataset_test = KGCRDataset(opt=opt_dataset_test, transform=transforms.Compose([ToTensor(opt_dataset_test)]))

    opt_model = {"n_entity": len(KGCR_dataset_train.entity2entityId), "n_relation": len(KGCR_dataset_train.relation2relationId), 
                "entity_embeddings": data_directory+"entity_embedding.txt", "dim": 300, "batch_size":32, "device": device, "lr": 1e-3,
                "epoch": 10, "model_directory": "../saved/models/", "model_name": "KGCR", "entity2entityId": KGCR_dataset_train.entity2entityId,
                "entity_embeddings": data_directory+"entity_embedding.txt", "movie_ids": KGCR_dataset_train.movie_ids}

    print(len(KGCR_dataset_train))
    
    KGCRDatasetLoaderTrain = DataLoader(KGCR_dataset_train, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0, collate_fn=pad_collate_entities)
    KGCRDatasetLoaderDev = DataLoader(KGCR_dataset_dev, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0, collate_fn=pad_collate_entities)
    KGCRDatasetLoaderTest = DataLoader(KGCR_dataset_test, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0, collate_fn=pad_collate_entities)

    KGCR_model_trainer = KGCRTrainer(opt_model)
    KGCR_model_trainer.train_model(KGCRDatasetLoaderTrain, KGCRDatasetLoaderDev)
    KGCR_model_trainer.evaluate_model(KGCRDatasetLoaderTest)

if __name__=="__main__":
    main()
