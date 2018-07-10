from utils import *
from functools import reduce

comps = load_components('../data/testid.txt')
users = generate_users(100)

MAX_FREQ = 10
NUM_RECORDS = 3000
NUM_PAIRS = 500

records = random_generate_records(NUM_RECORDS, NUM_PAIRS, users, comps, max_freq=MAX_FREQ)

ratio = reduce(lambda a, b: a + b, range(1, MAX_FREQ + 1)) / MAX_FREQ

print("预计生成纪录数：~{}".format(ratio * NUM_RECORDS))

users.to_csv('../data/users.csv')
records.to_csv('../data/records.csv')
decode_records(records, comps).to_csv('../data/records_decode.csv')
