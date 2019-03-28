import json
import joblib
import os
from numpy import zeros
from collections import Counter
from classes.embeddings.factory import EmbeddingsFactory

MAX_INPUT_SEQ_LENGTH = 25
MAX_VOCAB_SIZE = 50000
EMBEDDING_SIZE = 100

class Config(object):
    __config=dict()
    __X=None
    __Y=None
    def __init__(self, Txt_data,Txt_labels=None, max_input_seq_length=None,
                 max_vocab_size=None, num_target_tokens=2,
                 embedding=None,embedding_size=None,epochs=20 ):
        if max_input_seq_length is None:
            max_input_seq_length = MAX_INPUT_SEQ_LENGTH
        if max_vocab_size is None:
            max_vocab_size = MAX_VOCAB_SIZE
        if embedding_size is None:
            embedding_size=EMBEDDING_SIZE

        self.__X=Txt_data
        self.__Y=Txt_labels

        print("Config: creating word array ...")

        input_counter = Counter()
        max_seq_length = 0
        for line in Txt_data:
            text = [word.lower() for word in line.split(' ')]
            seq_length = len(text)
            if seq_length > max_input_seq_length:
                text = text[0:max_input_seq_length]
                seq_length = len(text)
            for word in text:
                input_counter[word] += 1
            max_seq_length = max(max_seq_length, seq_length)
        del input_counter['']
        word2idx = dict()

        for idx, word in enumerate(input_counter.most_common(max_vocab_size-2)):
            word2idx[word[0]] = idx + 2
        word2idx['PAD'] = 0
        word2idx['UNK'] = 1
        idx2word = dict([(idx, word) for word, idx in word2idx.items()])

        EmbeddingsFactory.path=Config.getPath('embedding')
        embedding_weights = None
        if embedding is not None:
            print("Config: Loading embeddings ...")
            if embedding.lower()=='glove':
                embedding_full=EmbeddingsFactory.getGloveEmbedding()

            e=embedding_full.popitem()
            embedding_size=e[1].__len__()

            embedding_full[e[0]]=e[1]
            embedding_weights=zeros((max_vocab_size,embedding_size))

            for word, idx in word2idx.items():
                embedding_vector = embedding_full.get(word)
                if embedding_vector is not None:
                    embedding_weights[idx] = embedding_vector

        self.__config = dict()
        self.__config['word2idx'] = word2idx
        self.__config['idx2word'] = idx2word
        self.__config['num_input_tokens'] = max_vocab_size
        self.__config['max_input_seq_length'] = max_input_seq_length
        self.__config['num_target_tokens']=num_target_tokens
        self.__config['embedding']=embedding
        self.__config['embedding_weights']=embedding_weights
        self.__config['embedding_size']=embedding_size
        self.__config['epochs'] = epochs

    @staticmethod
    def getPath(dirname):
        basedir=os.path.dirname(os.path.abspath(__file__))
        return {
            'data':basedir+'/Data/training',
            'models':basedir+'/models',
            'logs':basedir+'/logs',
            'reports':basedir+'/reports',
            'embedding':'x:/Embeddings',
        }.get(dirname,basedir+'/Data/training')

    def get(self,key):
        return self.__config.get(key)

    def set(self,key,val):
        self.__config[key]=val

    def to_json(self, **kwargs):
        return json.dumps(self.__config,skipkeys=True)
    def to_joblib(self,filename):
        joblib.dump(self.__config,filename)

    @staticmethod
    def from_joblib(filename):
        c = Config([''])
        c.__config=joblib.load(filename)
        return c
    @staticmethod
    def from_json(filename):
        c = Config([''])
        with open(filename, 'r') as f:
            c.__config= json.load(f)
        return c

    def getData(self):
        return self.__X
    def getLabels(self):
        return self.__Y