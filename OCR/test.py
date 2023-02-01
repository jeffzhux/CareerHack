import lmdb
import six
from PIL import Image

def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    im = Image.open(buf).convert(type)
    return im
env = lmdb.open('./data/train/MJ', max_readers= 1, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    label_key = b'label-%09d' % 223
    img_key = b'image-%09d' % 223
    
    # img = buf2PIL(txn, img_key, 'RGB')
    # img.save('hr.jpg')
    print(str(txn.get(label_key).decode()))
    print(str(txn.get(b'num-samples').decode()))
    # print(txn.get(b'image_hr-%09d' % 3))
