import itertools
from params import *

class Preprocessor():
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.vocab = []
        self.char_to_id = {}
        self.id_to_char = {}
        self.special_char = ['<pad>']

    def __call__(self, sentences):
        text = '\n'.join(sentences) 
        return self.transform(text)

    def fit(self, sentences: list):
        '''
        build dictionaries(char_to_id & id_to_char)

        Parameters
        ----------
            sentences: list
        
        Return
        ------
            vocab: 
                sorted vocabularies
            char_to_id: dict
                dictionary(key: char, val: id)
            id_to_char: dict
                dictionary(key: id, val: char)
        '''
        # データに現れる全ての文字をソートして取得
        self.vocab = sorted(set(list(itertools.chain.from_iterable(sentences))+['\n']))

        # 文字をキーにインデックスを返す辞書の作成
        # +1は予約語として<pad>を付け加えるため
        self.char_to_id = {v: idx+1 for idx, v in enumerate(list(self.vocab))}
        self.char_to_id['<pad>'] = 0
        
        # インデックスをキーに文字を返す辞書の作成
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
    
    def transform(self, text: str) -> list:
        '''
        translate text into the sequence of ids

        Parameters
        ----------
            text: str
                target text
            seq_length: int
                length of a sequence

        Return
        ------
            list of id list
        '''
        flat_ids = [self.char_to_id[c] for c in text]

        # 系列長で割り切れない場合はpaddingする
        if len(flat_ids) % self.seq_length != 0:
            rest_len = self.seq_length - (len(flat_ids) % self.seq_length)
            flat_ids += [0]*rest_len # id0は<pad>
        
        output = []
        for i in range(len(flat_ids) // self.seq_length):
            output.append(flat_ids[i*self.seq_length: (i+1)*self.seq_length])

        return output
    
    def decode(self, sentence_id: list) -> str:
        '''
        translate ids into the text

        Parameters
        ----------
            sentence_id: list
                the sequence of id
        
        Return
        ------
            text: str
        '''

        return ''.join([self.id_to_char[_id] for _id in sentence_id])