import lmdb, six, PIL
import numpy as np
from scipy.cluster.vq import *
from pylab import *
import json
from skimage import io
import os
import lmdb
import cv2
from tqdm import tqdm
import fire
from multiprocessing.pool import Pool
def clusterpixels(im, k):
    im = np.array(im)
    h, w = im.shape
    im = im.astype(np.float).reshape(-1)
    # 聚类， k是聚类数目
    centroids, variance = kmeans(im, k)
    code, distance = vq(im, centroids)
    code = code.reshape(h, w)
    fc = sum(code[:, 0])
    lc = sum(code[:, -1])
    fr = sum(code[0, :])
    lr = sum(code[-1, :])
    num = int(fr > w // 2) + int(lr > w // 2) + int(fc > h // 2) + int(lc > h // 2)
    if num >= 3:
        return 1 - code
    else:
        return code


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def write_lmdb(i, start, end, path):
    # root = '/data/TongkunGuan/data_lmdb_abinet/training/label/Synth/MJ/MJ_valid'
    root = path
    env = lmdb.open(
                root,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
    with env.begin(write=False) as txn:
        nSamples = int(txn.get("num-samples".encode()))

    # outputPath = f'/data/TongkunGuan/data_lmdb_abinet/Mask1/label/Synth/MJ/MJ_valid/{i}'
    outputPath = f"/media/xyw/831bebd9-c866-4ece-b878-5dbd68e5ca50/sjtu/GuanTongkun/Mask1/{root.replace('/home/xyw/sjtu/GuanTongkun/data_lmdb_abinet/training/','')}/{i}"
    os.makedirs(outputPath, exist_ok=True)
    create_env = lmdb.open(outputPath, map_size=21811627776)
    cache = {}
    cnt = 1

    with env.begin(write=False) as txn:
        for index in tqdm(range(start, end)):
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                image = PIL.Image.open(buf)
                image = image.convert("L")
                if np.array(image).shape[0] < 2 and np.array(image).shape[1] < 2:
                    print(f"Corrupted image for {index}")
                    continue
                mask = clusterpixels(image, 2)
                mask_Key = 'mask-%09d'.encode() % index
                # cache[mask_Key] = mask.astype(np.bool)
                cache[mask_Key] = cv2.imencode('.png', mask)[1].tobytes()
                if cnt % 1000 == 0:
                    writeCache(create_env, cache)
                    cache = {}
                    # print('Written %d / %d' % (cnt, nSamples))
                cnt += 1
            except IOError:
                print(f"Corrupted image for {index}")

        nSamples = cnt - 1
        cache['num-samples'.encode()] = str(nSamples).encode()
        writeCache(create_env, cache)
        print('Created dataset with %d samples' % nSamples)

# root = '/data/TongkunGuan/data_lmdb_abinet/training/label/Synth/MJ/MJ_valid'
root = ["/home/xyw/sjtu/GuanTongkun/data_lmdb_abinet/training/label/Synth",
        "/home/xyw/sjtu/GuanTongkun/data_lmdb_abinet/training/URD/OCR-CC"]
datasets=[]
def _get_dataset(paths):
    for p in paths:
        subfolders = [f.path for f in os.scandir(p) if f.is_dir()]
        if subfolders:  # Concat all subfolders
            _get_dataset(subfolders)
        else:
            datasets.append(p)
    return datasets
datasets = _get_dataset(root)

for path in datasets:
    env = lmdb.open(
                path,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
    with env.begin(write=False) as txn:
        nSamples = int(txn.get("num-samples".encode()))
    num_p = 36
    sub_num = nSamples // num_p
    p = Pool(num_p)
    args = []
    for i in range(num_p):
        if i == num_p -1:
            args.append((i, sub_num * i + 1, nSamples, path))
        else:
            args.append((i, sub_num * i + 1, sub_num * (i + 1) + 1, path))
    for arg in args:
        p.apply_async(write_lmdb, arg)

print('process end')


