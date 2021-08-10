import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import pandas as pd
from params import *
from dataset import HaikuDataset
from model import HaikuModel
from utils import preprocess, split_by_seqlength, text_to_ids


def train():
    # 俳句DataFrameの取得
    try:
        df = pd.read_pickle(HAIKU_PKL_PATH)
    except:
        import pickle
        with open(HAIKU_PKL_PATH, 'rb') as f:
            df = pickle.load(f)
    
    # 俳句リストの取得
    haiku_list = df.haiku.tolist()

    # vocab, char_to_id, id_to_charの取得
    vocab, char_to_id, id_to_char = preprocess(haiku_list)

    # 俳句リストを一つのテキストとして保持
    haiku_text = '\n'.join(haiku_list) 

    # haiku_textをidの列に変換
    ids = text_to_ids(haiku_text, char_to_id)

    # idsをSEQ_LENGTHで分割
    ids = split_by_seqlength(ids, SEQ_LENGTH)
    
    # 俳句データセットの作成
    dataset = HaikuDataset(ids)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # preparation for training
    vocab_size = len(vocab) + 1 # padding用のIDを追加するために+1をする
    model = HaikuModel(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train
    model.train()
    all_losses = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (X_train, y_train) in enumerate(dataloader):
            optimizer.zero_grad()
            X_train, y_train = X_train.to(model.device), y_train.to(model.device)
            state_h, state_c = model.initHidden(BATCH_SIZE)

            y_pred, (state_h, state_c) = model(X_train, (state_h, state_c))
            y_pred = y_pred.reshape(-1, vocab_size)
            y_train = y_train.reshape(-1).long()
            # print('y_pred shape: ', y_pred.shape)
            # print('y_train shape: ', y_train.shape)
            loss = criterion(y_pred, y_train)
            total_loss += loss

            loss.backward()
            optimizer.step()
        all_losses.append(total_loss / i)

        if epoch % 10 == 0:
            print(f'epoch: {epoch:3}, loss: {loss:3f}')
    


if __name__ == '__main__':
    train()