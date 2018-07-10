import keras
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from utils import *


class WordTable:

    def __init__(self, words):
        self.words = sorted(set(words))
        self.word_indices = {w: i + 1 for i, w in enumerate(self.words)}
        self.indices_word = {i + 1: w for i, w in enumerate(self.words)}

    def vocab_size(self):
        return len(self.words) + 1


class SimpleRNNRecommend:

    def __init__(self, user_maxlen: int
                 , comp_maxlen: int
                 , embed_hidden_size: int
                 , vocab_users: WordTable
                 , vocab_comps: WordTable
                 , lr=0.006):
        self.user_maxlen = user_maxlen
        self.comp_maxlen = comp_maxlen
        self.embed_hidden_size = embed_hidden_size
        self.vocab_users = vocab_users
        self.vocab_comps = vocab_comps
        self.lr = lr
        self.model = self.build_model(vocab_users, vocab_comps)

    def build_model(self, vocab_users: WordTable, vocab_comps: WordTable):
        RNN = recurrent.LSTM

        USER_MAXLEN = self.user_maxlen
        COMP_MAXLEN = self.comp_maxlen
        EMBED_HIDDEN_SIZE = self.embed_hidden_size

        user = layers.Input(shape=(USER_MAXLEN,), dtype=np.float32)
        encoded_user = layers.Embedding(vocab_users.vocab_size(), EMBED_HIDDEN_SIZE)(user)
        encoded_user = layers.Dropout(0.3)(encoded_user)

        comp = layers.Input(shape=(COMP_MAXLEN,), dtype=np.float32)
        encoded_comp = layers.Embedding(vocab_comps.vocab_size(), EMBED_HIDDEN_SIZE)(comp)
        encoded_comp = layers.Dropout(0.3)(encoded_comp)
        encoded_comp = RNN(EMBED_HIDDEN_SIZE)(encoded_comp)
        encoded_comp = layers.RepeatVector(USER_MAXLEN)(encoded_comp)

        merged = layers.add([encoded_user, encoded_comp])
        merged = RNN(EMBED_HIDDEN_SIZE)(merged)
        merged = layers.Dropout(0.3)(merged)

        preds = layers.Dense(vocab_comps.vocab_size(), activation='softmax')(merged)

        model = Model([user, comp], preds)

        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss=keras.losses.categorical_crossentropy
                      , metrics=['accuracy'])

        return model

    @staticmethod
    def load_data(records: pd.DataFrame):
        for i, row in records.iterrows():
            yield (row['userId'], row['compId'], row['followCompId'])

    @staticmethod
    def vectorize(data: iter
                  , comp_table: WordTable
                  , user_table: WordTable
                  , comp_maxlen: int
                  , user_maxlen: int):
        uc = []
        xc = []
        yc = []

        for user, comp, followComp in data:
            u = [user_table.word_indices[user]]
            x = [comp_table.word_indices[comp]]
            y = np.zeros(len(comp_table.words) + 1)
            y[comp_table.word_indices[followComp]] = 1
            uc.append(u)
            xc.append(x)
            yc.append(y)

        return pad_sequences(uc, maxlen=user_maxlen), pad_sequences(xc, maxlen=comp_maxlen), np.array(yc)

    def train(self, trains: pd.DataFrame, callbacks=None, epochs=1, batch_size=32):

        trains = list(self.load_data(trains))

        u, x, y = self.vectorize(trains, self.vocab_comps, self.vocab_users, self.comp_maxlen, self.user_maxlen)

        self.model.fit([u, x], y, callbacks=callbacks, batch_size=batch_size, validation_split=0.05, epochs=epochs)

    def evaluate(self, tests: pd.DataFrame, batch_size=32):

        tests = list(self.load_data(tests))

        tu, tx, ty = self.vectorize(tests, self.vocab_comps, self.vocab_users, self.comp_maxlen, self.user_maxlen)

        return self.model.evaluate([tu, tx], ty, batch_size=batch_size)

    def __save_weights(self, save_path):
        self.model.save_weights(save_path)

    def __save_vocab(self, vocab_dir):
        import pickle
        import os

        vocab_comps_path = os.path.join(vocab_dir, 'vocab_comps.vc')
        vocab_users_path = os.path.join(vocab_dir, 'vocab_users.vc')
        f = open(vocab_comps_path, 'wb')
        pickle.dump(self.vocab_comps, f)
        f.close()
        f = open(vocab_users_path, 'wb')
        pickle.dump(self.vocab_users, f)
        f.close()

    def __load_vocab(self, vocab_dir):
        import pickle
        import os

        vocab_comps_path = os.path.join(vocab_dir, 'vocab_comps.vc')
        vocab_users_path = os.path.join(vocab_dir, 'vocab_users.vc')

        f = open(vocab_comps_path, 'rb')
        self.vocab_comps = pickle.load(f)
        f.close()
        f = open(vocab_users_path, 'rb')
        self.vocab_users = pickle.load(f)
        f.close()

    def __load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def save_model(self, weights_path, vocab_dir):
        self.__save_weights(weights_path)
        self.__save_vocab(vocab_dir)

    def load_model(self, weights_path, vocab_dir):
        self.__load_vocab(vocab_dir)
        self.model = self.build_model(self.vocab_users, self.vocab_comps)
        self.__load_weights(weights_path)

    def predict(self, user: int, comp: int):
        u = pad_sequences([[self.vocab_users.word_indices[user]]])
        x = pad_sequences([[self.vocab_comps.word_indices[comp]]])
        return self.model.predict([u, x])
