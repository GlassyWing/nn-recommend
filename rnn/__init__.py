import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras import layers
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd

from utils import *

base_dir = '../data'
records_path = base_dir + '/records.csv'


records = pd.read_csv(records_path)

COMP_MAXLEN = 1
USER_MAXLEN = 1
EPOCHS = 4
BATCH_SIZE = 32


class WordTable:

    def __init__(self, words):
        self.words = sorted(set(words))
        self.word_indices = {w: i + 1 for i, w in enumerate(self.words)}
        self.indices_word = {i + 1: w for i, w in enumerate(self.words)}

    def vocab_size(self):
        return len(self.words) + 1


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


vocab_comps = WordTable(pd.concat([records['compId'], records['followCompId']], ignore_index=True))
vocab_users = WordTable(records['userId'])

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50

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

model.compile(optimizer=keras.optimizers.Adam(lr=0.006), loss=keras.losses.categorical_crossentropy
              , metrics=['accuracy'])

model.summary()


def load_data(records):
    for i, row in records.iterrows():
        yield (row['userId'], row['compId'], row['followCompId'])


train, test = train_test_split(list(load_data(records)), test_size=0.2)

u, x, y = vectorize(train, vocab_comps, vocab_users, COMP_MAXLEN, USER_MAXLEN)
tu, tx, ty = vectorize(test, vocab_comps, vocab_users, COMP_MAXLEN, USER_MAXLEN)

ck = ModelCheckpoint('../models/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss', verbose=0)

model.load_weights('weights.46-0.08.hdf5')


print('Training...')
model.fit([u, x], y,
          epochs=EPOCHS,
          validation_split=0.05, callbacks=[ck])

loss, acc = model.evaluate([tu, tx], ty,
                           batch_size=BATCH_SIZE)

print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
