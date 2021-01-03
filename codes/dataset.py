import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import load_pickle_file, _read_knowledge_graph, _edge_list
import functools
import operator

def pad_collate_entities(batch):
    currentEntities = []
    responseEntity = []

    # Process just the current utterance and the response
    for sample in batch:
        currentEntities.append(torch.LongTensor(sample[0]))
        responseEntity.append(sample[1])

    currentEntitiesPadded = pad_sequence(currentEntities, batch_first=True, padding_value=0)
    currentEntitiesMask = currentEntitiesPadded != 0

    responseEntity = torch.LongTensor(responseEntity)

    batch = {"currentEntities": currentEntities, "currentEntitiesMask": currentEntitiesMask, "responseEntity": responseEntity}

    return batch

class KGCRDataset(Dataset):
    def __init__(self, opt, transform):
        self.transform = transform
        self.entity2entityId = load_pickle_file(opt["entity2entityId"])
        self.relation2relationId = load_pickle_file(opt["relation2relationId"])
        self.dataset = load_pickle_file(opt["dataset"])
        self.movie_ids = load_pickle_file(opt["movie_ids"])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToTensor(object):
    """Convert the entities to indexes.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, opt):
        self.entity2entityId = load_pickle_file(opt["entity2entityId"])
        self.relation2relationId = load_pickle_file(opt["relation2relationId"])

    def __call__(self, sample):
        MovieEntities = sample[0][3][1].split(" ")
        MentionedEntities = sample[0][3][3].split(" ")
        responseEntity = sample[0][3][2]
        
        MovieEntities = [int(entity) for entity in MovieEntities if len(entity)]
        MentionedEntities = [int(entity) for entity in MentionedEntities if len(entity)]
        responseEntity = int(responseEntity)
        
        return [MovieEntities+MentionedEntities, responseEntity]
