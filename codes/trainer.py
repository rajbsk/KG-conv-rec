import copy
import os
import pickle as pkl
import re
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import nltk
from tqdm import tqdm

from KG_conv_rec_model import KGCR
from utils import _load_kg_embeddings
from logger import Logger

class KGCRTrainer():
    def __init__(self, opt):
        self.n_entity = opt["n_entity"]
        self.device = opt["device"]
        self.entity2entityId = opt["entity2entityId"]
        self.epochs = opt["epoch"]
        entity_kg_emb = _load_kg_embeddings(self.entity2entityId, opt["dim"], opt["entity_embeddings"])

        # encoder captures the input text
        self.model = KGCR(n_entity=opt["n_entity"], n_relation=opt["n_relation"], dim=opt["dim"], entity_kg_emb=entity_kg_emb)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),opt["lr"])

        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)
        self.reset_metrics()

    def report(self):
        """
        Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        base = super().report()
        m = {}
        m["num_tokens"] = self.counts["num_tokens"]
        m["num_batches"] = self.counts["num_batches"]
        m["loss"] = self.metrics["loss"] / m["num_batches"]
        m["base_loss"] = self.metrics["base_loss"] / m["num_batches"]
        m["acc"] = self.metrics["acc"] / m["num_tokens"]
        m["auc"] = self.metrics["auc"] / m["num_tokens"]
        # Top-k recommendation Recall
        for x in sorted(self.metrics):
            if x.startswith("recall") and self.counts[x] > 200:
                m[x] = self.metrics[x] / self.counts[x]
                m["num_tokens_" + x] = self.counts[x]
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key] = 0.0
        for key in self.counts:
            self.counts[key] = 0

    def process_batch(self, batch):
        entities = [entities.to(self.device) for entities in (batch["currentEntities"])]
        labels = (batch["responseEntity"]).to(self.device)
        return entities, labels

    def train_step(self, batch):
        self.model.train()
        bs = len(batch)
        entities, labels = self.process_batch(batch)

        return_dict = self.model(entities, labels)

        loss = return_dict["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metrics["base_loss"] += return_dict["base_loss"].item()
        self.metrics["loss"] += loss.item()

        self.counts["num_tokens"] += bs
        self.counts["num_batches"] += 1
        return loss.item()

    def eval_step(self, batch):
        self.model.eval()
        bs = (batch.label_vec == 1).sum().item()
        labels = torch.zeros(bs, dtype=torch.long)

        # create subgraph for propagation
        bs = len(batch)
        entities, labels = self.process_batch(batch)

        return_dict = self.model(entities, labels)

        loss = return_dict["loss"]

        self.metrics["base_loss"] += return_dict["base_loss"].item()
        self.metrics["loss"] += loss.item()
        self.counts["num_tokens"] += bs
        self.counts["num_batches"] += 1

        outputs = return_dict["scores"].cpu()
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=outputs.size()[1], dim=1)
        for b in range(bs):
            target_idx = self.movie_ids.index(labels[b].item())
            rank = (pred_idx[b] == target_idx).nonzero().tolist()[0][0]
            self.metrics["recall@100"] += 1.0/(1+rank)
            self.metrics["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            self.metrics[f"recall@1@turn{turns[b]}"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics[f"recall@10@turn{turns[b]}"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics[f"recall@50@turn{turns[b]}"] += int(target_idx in pred_idx[b][:50].tolist())
            self.counts[f"recall@1@turn{turns[b]}"] += 1
            self.counts[f"recall@10@turn{turns[b]}"] += 1
            self.counts[f"recall@50@turn{turns[b]}"] += 1
            self.counts[f"recall@1"] += 1
            self.counts[f"recall@10"] += 1
            self.counts[f"recall@50"] += 1
            self.counts[f"recall@MRR"] += 1
    
    def evaluate_model(self, dataLoader):
        self.reset_metrics()
        total_loss = 0
        for batch in dataLoader:
            batch_loss = self.eval_step(batch, train=False)
            total_loss += batch_loss
        print("Evaluation Loss: %f"%(total_loss))
        
        return total_loss
    
    def train_model(self, trainDataLoader, devDataLoader):
        logger = Logger("../logger/")
        self.optimizer.zero_grad()
        ins=0
        for epoch in range(self.epochs):
            self.reset_metrics()
            train_loss = 0
            for idx, batch in tqdm(enumerate(trainDataLoader)):
                batch_loss = self.train_step(batch)
                train_loss += batch_loss
            dev_loss = self.evaluate_model(devDataLoader)
            print("Iteration: %d,Train Loss = %f" %(epoch, train_loss))
            
            # Logging parameters
            p = list(self.named_parameters())
            logger.scalar_summary("Train Loss", train_loss, ins+1)
            logger.scalar_summary("Dev Loss", dev_loss, ins+1)
            for tag, value in self.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), ins+1)
                if value.grad != None:
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), ins+1)
            ins+=1
            if (epoch+1)%self.save_every==0:
                torch.save(self.state_dict(), self.model_directory+self.model_name+"_"+str(epoch+1))

