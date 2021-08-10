import pandas as pd
import numpy as np
from params import *
import itertools

def preprocess(haiku_list: list):
    '''
    preprocess for haiku list
    
    Parameters
    ----------
        haiku_list: list
            list of haiku
    
    Return
    ------
        vocab: set of vocabularies
        char_to_id: translate chars to ids
        id_to_char: translate ids to chars
    '''
    vocab = sorted(set(list(itertools.chain.from_iterable(haiku_list))+['\n']))

    # 文字をキーにインデックスを返す辞書の作成
    char_to_id = {}
    for idx, v in enumerate(list(vocab)):
        char_to_id[v] = idx
    
    # インデックスをキーに文字を返す辞書の作成
    id_to_char = {v:k for k, v in char_to_id.items()}

    return vocab, char_to_id, id_to_char

def text_to_ids(text: str, char_to_id: dict) -> list:
    '''
    translate text into the sequence of ids

    Parameters
    ----------
        text: str
            target text
        char_to_id: dict
            char - id dictionary 

    Return
    ------
        list of ids
    '''
    return [char_to_id[c] for c in text]

def split_by_seqlength(ids: list, seq_length: int):
    '''
    split the sequence of ids by the passed seq_length

    Parameters
    ----------
        ids: list
            the sequence of ids
        seq_length: int
            the number of seqence length

    Return
    ------
        numpy array of id list
    '''
    np_ids = np.array(ids)
    if np_ids.shape[0] % seq_length != 0:
        # seq_lengthの倍数出ない場合は、0で埋める
        diff = seq_length * (np_ids.shape[0] // seq_length + 1) - np_ids.shape[0]
        np_ids = np.append(np_ids, [-1] * diff) # padding idとして使用されていない-1を用いる
    np_ids = np_ids.reshape([-1, seq_length])
    
    return np_ids
    

def get_input_target(text: str):
    '''
    
    Parameters
    ----------
        text: str

    Return
    ------
        input_text, target_text
    '''
    num_batches = int(len(text) / (SEQ_LENGTH * BATCH_SIZE))
    input_text = text[:num_batches*SEQ_LENGTH*BATCH_SIZE]
    target_text = np.zeros_like(input_text)
    target_text[:-1] = input_text[1:]
    target_text[-1]  = input_text[0]
    input_text  = np.reshape(input_text, (BATCH_SIZE, -1))
    target_text = np.reshape(target_text, (BATCH_SIZE, -1))

    return input_text, target_text