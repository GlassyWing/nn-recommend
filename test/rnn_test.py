from unittest import TestCase

from keras.callbacks import ModelCheckpoint

from tools.decode import *
from rnn import WordTable, SimpleRNNRecommend
import pandas as pd
from keras import backend as K

USER_MAXLEN = 1
COMP_MAXLEN = 1
EMBED_HIDDEN_UNITS = 50
EPOCHS = 5
BATCH_SIZE = 32

base_dir = '../data'
records_path = base_dir + '/records.csv'

records = pd.read_csv(records_path)

# 混淆数据
records = records.sample(frac=1).reset_index(drop=True)

# 按照8：2分割为训练集和测试集
split_pot = int(len(records) * 0.8)
trains = records[:split_pot]
tests = records[split_pot:]

# 词汇表
vocab_users = WordTable(records['userId'])
vocab_comps = WordTable(pd.concat([records['compId'], records['followCompId']], ignore_index=True))


class RNNTest(TestCase):

    def setUp(self):
        model = SimpleRNNRecommend(USER_MAXLEN, COMP_MAXLEN, EMBED_HIDDEN_UNITS, vocab_users, vocab_comps)
        self.model = model

    def test_model_train(self):
        """
        训练模型，迭代5次，并将权重保存到文件中
        :return:
        """
        ck = ModelCheckpoint('../models/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss', verbose=0)

        self.model.train(trains, callbacks=[ck], batch_size=32, epochs=5)
        loss, acc = self.model.evaluate(tests)

        print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

    def test_model_predict(self):
        weighs_path = '../models/weights.05-0.28.hdf5'
        self.model.load_weights(weighs_path)
        preds = self.model.predict(1, 68, vocab_users, vocab_comps)
        for comp, possibility in next_comps(preds, 0.5, vocab_comps):
            print("Next comp: {}, possibility: {}".format(comp, possibility))
