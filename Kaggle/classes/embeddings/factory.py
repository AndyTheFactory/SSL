import time

import numpy as np
import os
import json
import joblib

GLOVE_FILE=r"\glove.840B.300d\glove.840B.300d.txt"
GOOGLENEWS_FILE=r"\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin"
PARAGRAM_FILE=r"\paragram_300_sl999\paragram_300_sl999.txt"
WIKI_FILE=r"\wiki-news-300d-1M\wiki-news-300d-1M.vec"



class EmbeddingsFactory(object):
    path=''
    embeddings={}
    @staticmethod
    def getGloveEmbedding():
        if 'glove' in EmbeddingsFactory.embeddings:
            return EmbeddingsFactory.embeddings['glove']
        # if os.path.isfile(EmbeddingsFactory.path+'/glove.joblib'):
        #     EmbeddingsFactory.embeddings['glove']=joblib.load(EmbeddingsFactory.path+'/glove.joblib')
        #     return EmbeddingsFactory.embeddings['glove']

        print("loading glove embeddings")
        _word2em = {}
        glove_model_path = EmbeddingsFactory.path+GLOVE_FILE
        file = open(glove_model_path, mode='rt', encoding='utf8')
        for line in file:
            words = line.strip().split(" ")
            word = words[0]
            embeds = np.array(words[1:], dtype=np.float32)
            _word2em[word] = embeds
        file.close()
        EmbeddingsFactory.embeddings['glove']=_word2em
        # joblib.dump(_word2em,filename=EmbeddingsFactory.path+'/glove.joblib')

        return EmbeddingsFactory.embeddings['glove']

    @staticmethod
    def getParagramEmbedding():
        if 'paragram' in EmbeddingsFactory.embeddings:
            return EmbeddingsFactory.embeddings['paragram']
        # if os.path.isfile(EmbeddingsFactory.path+'/glove.joblib'):
        #     EmbeddingsFactory.embeddings['glove']=joblib.load(EmbeddingsFactory.path+'/glove.joblib')
        #     return EmbeddingsFactory.embeddings['glove']

        print("loading paragram embeddings")
        _word2em = {}
        glove_model_path = EmbeddingsFactory.path+PARAGRAM_FILE
        file = open(glove_model_path, mode='rt', encoding='utf8')
        for line in file:
            words = line.strip().split(" ")
            word = words[0]
            embeds = np.array(words[1:], dtype=np.float32)
            _word2em[word] = embeds
        file.close()
        EmbeddingsFactory.embeddings['paragram']=_word2em
        # joblib.dump(_word2em,filename=EmbeddingsFactory.path+'/glove.joblib')

        return EmbeddingsFactory.embeddings['paragram']

    @staticmethod
    def getWikiEmbedding():
        if 'wiki' in EmbeddingsFactory.embeddings:
            return EmbeddingsFactory.embeddings['wiki']
        # if os.path.isfile(EmbeddingsFactory.path+'/glove.joblib'):
        #     EmbeddingsFactory.embeddings['glove']=joblib.load(EmbeddingsFactory.path+'/glove.joblib')
        #     return EmbeddingsFactory.embeddings['glove']

        print("loading wiki embeddings")
        _word2em = {}
        glove_model_path = EmbeddingsFactory.path+WIKI_FILE
        file = open(glove_model_path, mode='rt', encoding='utf8')
        for line in file:
            words = line.strip().split(" ")
            word = words[0]
            embeds = np.array(words[1:], dtype=np.float32)
            _word2em[word] = embeds
        file.close()
        EmbeddingsFactory.embeddings['wiki']=_word2em
        # joblib.dump(_word2em,filename=EmbeddingsFactory.path+'/glove.joblib')

        return EmbeddingsFactory.embeddings['wiki']

