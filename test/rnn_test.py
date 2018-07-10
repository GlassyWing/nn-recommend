from unittest import TestCase

from keras.callbacks import ModelCheckpoint

from decode import *
from utils import *
from rnn import WordTable, SimpleRNNRecommend
import pandas as pd

USER_MAXLEN = 1
COMP_MAXLEN = 1
EMBED_HIDDEN_UNITS = 50
EPOCHS = 30
BATCH_SIZE = 32

base_dir = '../data'
records_path = base_dir + '/records.csv'
comp_path = base_dir + '/testid.txt'

comps = load_components(comp_path)
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
        model = SimpleRNNRecommend(USER_MAXLEN, COMP_MAXLEN, EMBED_HIDDEN_UNITS
                                   , vocab_users
                                   , vocab_comps
                                   , lr=0.007)
        self.model = model

    def test_model_train(self):
        """
        训练模型，迭代30次，并将权重保存到文件中
        :return:
        """
        ck = ModelCheckpoint('../models/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss', verbose=0)

        self.model.train(trains, callbacks=[ck], batch_size=BATCH_SIZE, epochs=EPOCHS)

        loss, acc = self.model.evaluate(tests)

        print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

        # 保存训练好的模型
        self.model.save_model('../models/weights.final.hdf5', '../models')

    def test_model_predict(self):
        """
        测试使用模型进行预测
        :return:
        """
        weighs_path = '../models/weights.final.hdf5'
        vocab_dir = '../models'

        # 载入模型
        self.model.load_model(weighs_path, vocab_dir)

        vocab_users = self.model.vocab_users
        vocab_comps = self.model.vocab_comps

        userId = 0

        # 通过构件名获得构件Id
        # comp_name = 'JSONUTIL.get'
        # compId = comps[comps['name'] == comp_name]['compId'].values[0]

        compId = 669

        # 获得下一个可能构件的概率分布
        import time
        start = time.time()
        preds = self.model.predict(userId, compId)
        end = time.time()
        print("Predict cost {} s".format(end - start))

        # 将概率分布转换为可读的表示
        for next_comp_id, possibility in next_comps(preds, 0.06, vocab_comps):
            next_comp = comps[comps['compId'] == next_comp_id]['name'].values[0]
            print("Next comp id: {}, name: {}, possibility: {}".format(next_comp_id, next_comp, possibility))
