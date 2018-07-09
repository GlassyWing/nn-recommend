from utils import *

comps = load_components('../data/testid.txt')
users = generate_users(100)

records = random_generate_records(5 * 100 ** 2, 1 * 100 ** 2, users, comps)
users.to_csv('../data/users.csv')
records.to_csv('../data/records.csv')
decode_records(records, comps).to_csv('../data/records_decode.csv')