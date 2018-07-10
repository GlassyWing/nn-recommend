import pandas as pd
import numpy as np
import random


def load_components(comp_path) -> pd.DataFrame:
    """
    载入构件
    :param comp_path: 构件文件路径
    :return:
    """
    df = pd.read_csv(comp_path, dtype={'compId': np.int32, 'name': str})
    return df


def generate_users(num_user) -> pd.DataFrame:
    """
    生成用户
    :param num_user: 用户数量
    :return:
    """
    df = pd.DataFrame(np.arange(0, num_user), index=np.arange(0, num_user), columns=['userId'])
    return df


def __generate_comp_pairs(comps: pd.DataFrame, num_pairs):
    """
    随机生成构件对
    :param comps: 所有构件
    :param num_pairs: 构件对的数量
    :return:
    """
    for i in range(num_pairs):
        comp_id_01 = random.choice(comps['compId'])
        comp_id_02 = random.choice(comps['compId'])
        yield (comp_id_01, comp_id_02)


def random_generate_records(num_records: int, num_pairs: int, users: pd.DataFrame, comps: pd.DataFrame,
                            max_freq=10) -> pd.DataFrame:
    """
    随机生成构件使用纪录
    :param max_freq:
    :param num_records: 纪录条数
    :param num_pairs: 构件对的数量
    :param users: 用户
    :param comps: 构件
    :return:
    """
    comp_pairs = list(__generate_comp_pairs(comps, num_pairs))

    records = []

    for i in range(num_records):
        comp_pair = random.choice(comp_pairs)
        user_id = random.choice(users.index)
        freq = random.randint(1, max_freq)
        for _ in range(freq):
            records.append((user_id, comp_pair[0], comp_pair[1]))

    return pd.DataFrame(records, columns=['userId', 'compId', 'followCompId']).sample(frac=1).reset_index(drop=True)


def decode_records(records: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
    """
    将纪录中的构件Id替换为构件名
    :param records:
    :param comps:
    :return:
    """
    stage = pd.merge(records, comps, on='compId')
    stage = pd.merge(stage, comps, left_on='followCompId', right_on='compId', how='left') \
        .drop(columns=['compId_y', 'compId_x', 'followCompId']) \
        .rename(columns={'name_x': 'comp_name', 'name_y': 'follow_comp_name'})
    return stage
