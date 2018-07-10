from utils import *
from functools import reduce

"""
该文件用于随机生成纪录
"""

comps = load_components('data/testid.txt')
users = generate_users(100)

MAX_FREQ = 10
NUM_RECORDS = 15000
NUM_PAIRS = 1000

records = random_generate_records(NUM_RECORDS, NUM_PAIRS, users, comps, max_freq=MAX_FREQ)

users.to_csv('data/users.csv')
records.to_csv('data/records.csv')
decode_records(records, comps).to_csv('data/records_decode.csv')
