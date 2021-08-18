import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from params import *
from preprocessor import Preprocessor
from model import HaikuModel
from filters import HaikuFilter

def generate():
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
    prepro = Preprocessor(SEQ_LENGTH)
    prepro.fit(haiku_list)

    # modelの構築
    VOCAB_SIZE = len(prepro.char_to_id)
    model = HaikuModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS)
    
    # modelのload
    model.load_state_dict(torch.load('../checkpoints/ckpt_40.pt')['model_state_dict'])
    model.eval()

    # 俳句フィルター
    haiku_filter = HaikuFilter()

    # inference
    generated_haikulist = []
    next_char = "\n"
    with torch.no_grad():
        # 指定した回数だけ俳句を生成する
        while len(generated_haikulist) < HAIKU_NUM:
            states = model.initHidden(batch_size=1) # inference時のbatch sizeは1
            haiku = ''

            # 俳句の生成
            while True:
                input_id = [[prepro.char_to_id[next_char]]]
                input_tensor = torch.tensor(input_id, device=model.device)
                logits, states = model(input_tensor, states)
                probs = F.softmax(torch.squeeze(logits)).cpu().detach().numpy()
                next_id = np.random.choice(VOCAB_SIZE, p=probs)
                next_char = prepro.id_to_char[next_id]

                # 改行が出たら俳句の完成合図
                if next_char == '\n':
                    break
                else:
                    haiku += next_char

            # 俳句のフィルタリング

            # 1. 季語の有無チェック
            kigo, season = haiku_filter.check_kigo(haiku)
            if not kigo: # 季語ない場合は俳句として認めない
                continue

            # フィルターをパスした句はgenerated_haikulistに追加する
            generated_haikulist.append(haiku)
            print(haiku)

if __name__=="__main__":
    generate()