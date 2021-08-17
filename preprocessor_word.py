import itertools
from params import *
import MeCab

class Preprocessor():
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.vocab = []
        self.char_to_id = {}
        self.id_to_char = {}
        self.special_char = ['<pad>']
        self.tagger = MeCab.Tagger('-Owakati')

    def __call__(self, sentences):
        morphemes = [] # 形態素のリスト
        for sentence in sentences:
            node = self.tagger.parseToNode(sentence)
            while node:
                features = node.feature.split(',')
                if features[0] != u'BOS/EOS':
                    morphemes.append(node.surface)
                node = node.next
            morphemes.append('\n') # 俳句の終わりの合図として改行を挿入する

        return self.transform(morphemes)

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
        # テキストデータに現れる全ての形態素を取得する
        morphemes = []
        for sentence  in sentences:
            node = self.tagger.parseToNode(sentence)
            while node:
                features = node.feature.split(',')
                if features[0] != u'BOS/EOS':
                    morphemes.append(node.surface)
                node = node.next
        
        # ソートしてvocabularyとして保持する
        self.vocab = sorted(set(morphemes+['\n']))

        # 文字をキーにインデックスを返す辞書の作成
        # +1は予約語として<pad>を付け加えるため
        self.char_to_id = {v: idx+1 for idx, v in enumerate(list(self.vocab))}
        self.char_to_id['<pad>'] = 0
        
        # インデックスをキーに文字を返す辞書の作成
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
    
    def transform(self, morphemes: list) -> list:
        '''
        translate morpheme list into the sequence of ids

        Parameters
        ----------
            morphemese: list
                the sequence of morpheme

        Return
        ------
            list of id list
        '''
        flat_ids = [self.char_to_id[morpheme] for morpheme in morphemes]

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