from os.path import join

from keras.models import Sequential
from keras.layers import Embedding, Dense, SpatialDropout1D, Bidirectional
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
import numpy as np

EMBEDDING_SIZE = 100
BATCH_SIZE = 64
VERBOSE = 1
EPOCHS = 30


def generate_batch(x_samples, y_samples):
    num_batches = len(x_samples) // BATCH_SIZE

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            yield x_samples[start:end], y_samples[start:end]


class BiLstmClassifier(object):
    model_name = 'bilstm'

    def __init__(self, config,lstm_units=64,dropout=0.2,embedding_size=None):
        self.num_input_tokens = config.get('num_input_tokens')
        self.max_input_seq_length = config.get('max_input_seq_length')
        self.num_target_tokens = config.get('num_target_tokens')
        self.word2idx = config.get('word2idx')
        self.idx2word = config.get('idx2word')
        self.config = config

        embedding_size = config.get('embedding_size')

        model = Sequential()
        if config.get('embedding') is None:
            model.add(Embedding(input_dim=self.num_input_tokens, output_dim=embedding_size,
                                name="embedding",
                                input_length=self.max_input_seq_length))
        else:
            model.add(Embedding(input_dim=self.num_input_tokens, output_dim=embedding_size,
                                name="embedding",
                                input_length=self.max_input_seq_length,
                                weights=[config.get('embedding_weights')],
                                trainable=False))
        model.add(SpatialDropout1D(dropout))
        model.add(Bidirectional(LSTM(units=lstm_units, dropout=dropout, recurrent_dropout=dropout)))
        model.add(Dense(self.num_target_tokens, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def load_weights(self, weight_file_path):
        self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.word2idx:
                    wid = self.word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        #print(temp.shape)
        return temp

    def transform_target_encoding(self, targets):
        return np_utils.to_categorical(targets, num_classes=self.num_target_tokens)

    def fit(self, Xtrain, Ytrain, Xtest=None, Ytest=None, epochs=None, file_prefix=None):
        if epochs is None:
            epochs = EPOCHS

        model_dir_path = self.config.getPath('models')

        filename = self.model_name
        if file_prefix != None:
            filename = file_prefix+'_'+self.model_name

        filename = model_dir_path + '/' + filename

        # save model weights
        config_file_path = filename + '-config.joblib'
        weight_file_path = filename + '-weights.h5'

        checkpoint = ModelCheckpoint(weight_file_path)

        self.config.to_joblib(config_file_path)

        # save model architecture
        architecture_file_path = filename + '-architecture.json'
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        if Ytest is not None:
            Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        if Xtest is not None:
            Xtest = self.transform_input_text(Xtest)

        train_gen = generate_batch(Xtrain, Ytrain)
        if (Xtest is not None)and(Ytest is not None):
            test_gen = generate_batch(Xtest, Ytest)
            test_num_batches = len(Xtest) // BATCH_SIZE
        else:
            test_gen=None
            test_num_batches = 0 // BATCH_SIZE

        train_num_batches = len(Xtrain) // BATCH_SIZE

        with open(join(self.config.getPath('logs'), 'metadata.tsv'), 'w') as f:
            np.savetxt(f, Ytest)
            f.close()

        tensorboard = TensorBoard(log_dir=self.config.getPath('logs'),
                                  batch_size=BATCH_SIZE,
                                  embeddings_freq=1,
                                  embeddings_layer_names=['embedding'],
                                  update_freq='batch',
                                  histogram_freq=1,
                                  write_graph=True,
                                  embeddings_metadata=join(self.config.getPath('logs'), 'metadata.tsv'),
                                  embeddings_data=Xtest)


        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def predict(self, x):
        is_str = False
        if type(x) is str:
            is_str = True
            x = [x]

        Xtest = self.transform_input_text(x)

        preds = self.model.predict(Xtest)
        if is_str:
            preds = preds[0]
            return np.argmax(preds)
        else:
            return np.argmax(preds, axis=1)

