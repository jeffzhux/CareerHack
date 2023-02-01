
import lmdb
env = lmdb.open('./OCR/data/valid')

index = 151
with env.begin(write=False) as txn:
    label_key = 'label-%09d'.encode() % index
    # label = txn.get(label_key).decode('utf-8')
    label = txn.get(label_key).decode()
    print(label)