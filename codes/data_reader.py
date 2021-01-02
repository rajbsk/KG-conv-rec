import copy
import re
import csv
import json
import os
import pickle as pkl

import requests

from utils import load_pickle_file

def _text2entities(text, text_dict):
    return text_dict[text]

def _convert_ids_to_indices(text, questions, id2entity, entity2entityId):
    """@movieID -> @movieIdx"""
    pattern = re.compile("@\d+")
    movieId_list = []

    def convert(match):
        movieId = match.group(0)
        try:
            entity = id2entity[int(movieId[1:])]
            if entity is not None:
                movieId_list.append(str(entity2entityId[entity]))
            else:
                movieId_list.append(str(entity2entityId[int(movieId[1:])]))
        except Exception:
            return ""

    return re.sub(pattern, convert, text), movieId_list

def _get_entities(text, text_dict, entity2entityId):
    """text -> [#entity1, #entity2]"""
    entities = _text2entities(text, text_dict)
    entities = [str(entity2entityId[x]) for x in entities if x in entity2entityId]
    return entities

def read_data(path, text_dict, entity2entityId, id2entity):
    instances = []
    with open(path) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))
    dataset = []
    # define iterator over all queries
    for instance in instances:
        initiator_id = instance["initiatorWorkerId"]
        respondent_id = instance["respondentWorkerId"]
        messages = instance["messages"]
        message_idx = 0
        new_episode = True
        dialog_text = ""
        previously_mentioned_movies_list = []
        mentioned_entities = []
        turn = 0
        while message_idx < len(messages):
            source_text = []
            target_text = []
            while (
                message_idx < len(messages)
                and messages[message_idx]["senderWorkerId"] == initiator_id
            ):
                source_text.append(messages[message_idx]["text"])
                message_idx += 1
            while (
                message_idx < len(messages)
                and messages[message_idx]["senderWorkerId"] == respondent_id
            ):
                target_text.append(messages[message_idx]["text"])
                message_idx += 1
            source_text = [text for text in source_text if text != ""]
            target_text = [text for text in target_text if text != ""]
            if source_text != [] or target_text != []:
                for src in source_text:
                    mentioned_entities += _get_entities(src, text_dict, entity2entityId)
                target_mentioned_entities = []
                for tgt in target_text:
                    target_mentioned_entities += _get_entities(tgt, text_dict, entity2entityId)
                source_text = '\n'.join(source_text)
                target_text = '\n'.join(target_text)
                dialog_text += source_text[:] + "\t"
                source_text, source_movie_list = _convert_ids_to_indices(
                    source_text, instance["initiatorQuestions"], id2entity, entity2entityId
                )
                target_text, target_movie_list = _convert_ids_to_indices(
                    target_text, instance["initiatorQuestions"], id2entity, entity2entityId
                )
                turn += 1
                if len(target_movie_list):
                    for target_movie in target_movie_list:
                        dataset.append(((source_text, [target_text], None, [str(turn), ' '.join(previously_mentioned_movies_list + source_movie_list), target_movie, ' '.join(mentioned_entities), target_text], None), new_episode))
                new_episode = False
                dialog_text += target_text[:] + "\t"
                previously_mentioned_movies_list += source_movie_list + target_movie_list
                mentioned_entities += target_mentioned_entities
    return dataset

def main():
    datapath = "../dataset/data_2/redial/"
    train_file = datapath + "train_data.jsonl"
    test_file = datapath + "test_data.jsonl"
    valid_file = datapath + "valid_data.jsonl"

    entity2entityId = load_pickle_file(datapath+"entity2entityId.pkl")
    text_dict = load_pickle_file(datapath+"text_dict.pkl")
    id2entity = load_pickle_file(datapath+"id2entity.pkl")

    train_dataset = read_data(train_file, text_dict, entity2entityId, id2entity)
    test_dataset = read_data(test_file, text_dict, entity2entityId, id2entity)
    valid_dataset = read_data(valid_file, text_dict, entity2entityId, id2entity)

    with open(datapath + "dataset_train.pkl", "wb") as f:
        pkl.dump(train_dataset, f)
    with open(datapath + "dataset_test.pkl", "wb") as f:
        pkl.dump(test_dataset, f)
    with open(datapath + "dataset_valid.pkl", "wb") as f:
        pkl.dump(valid_dataset, f)


if __name__ == "__main__":
    main()
