import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import pandas as pd
import numpy as np
from params import *
from preprocessor import Preprocessor
from dataset import HaikuDataset
from model import HaikuModel


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

    # preprocessor 
    haiku_preprocessor = Preprocessor(SEQ_LENGTH)
    haiku_preprocessor.fit(haiku_list)
    
    # haiku_textをidの列に変換
    ids_list = haiku_preprocessor(haiku_list)

    # 俳句データセットの作成
    dataset = HaikuDataset(ids_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # trainingの準備
    vocab_size = len(haiku_preprocessor.char_to_id)
    model = HaikuModel(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # checkpointsディレクトリの作成
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tar_dir = os.path.join(cur_dir, 'checkpoints')
    os.makedirs(tar_dir, exist_ok=True)

    # train
    model.train()
    all_losses = []
    for epoch in range(EPOCHS):
        print('-'*25)
        print(f'EPOCH: {epoch+1}')

        total_loss = 0
        for X_train, y_train in dataloader:
            optimizer.zero_grad()
            X_train, y_train = X_train.to(model.device), y_train.to(model.device)
            state_h, state_c = model.initHidden(BATCH_SIZE)

            y_pred, (state_h, state_c) = model(X_train, (state_h, state_c))
            y_pred = y_pred.view(-1, vocab_size)
            y_train = y_train.view(-1)
            loss = criterion(y_pred, y_train)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        current_loss = total_loss / len(dataloader)
        current_ppl  = np.exp(current_loss)
        print(f'LOSS: {current_loss}, PERPLEXITY: {current_ppl}')
        all_losses.append(current_loss)

        # 10epoch毎にsave
        if epoch % 10 == 0:
            path = f'./checkpoints/ckpt_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss
            }, path)
    
    # 学習の最後にモデルを保存
    torch.save(model.state_dict(), f'./checkpoints/final.pt')

if __name__ == '__main__':
    train()