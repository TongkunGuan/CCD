import lmdb
from tqdm import tqdm
# 将两个lmdb文件合并成一个新的lmdb
num_p = 36
# env代表Environment, txn代表Transaction
root = '/data/TongkunGuan/data_lmdb_abinet/Mask1/label/Synth/MJ/MJ_valid'
# 打开lmdb文件，写模式，
env_3 = lmdb.open(root, map_size=int(1e12))
txn_3 = env_3.begin(write=True)
for i in range(num_p):
    lmdb_path = root + f'/{i}'
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    txn = env.begin()
    # 打开数据库
    database = txn.cursor()
    count = 0
    # 遍历数据库
    for (key, value) in tqdm(database):
        # 将数据放到结果数据库事务中
        txn_3.put(key, value)
        count += 1
        if count % 1000 == 0:
            # 将数据写入数据库，必须的，否则数据不会写入到数据库中
            txn_3.commit()
            txn_3 = env_3.begin(write=True)
    txn_3.commit()
    txn_3 = env_3.begin(write=True)
    # 关闭lmdb
    env.close()
# 输出结果lmdb的状态信息，可以看到数据是否合并成功
print(env_3.stat())
env_3.close()