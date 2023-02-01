import lmdb
import csv
import cv2

def img2byte(img):
    is_success, img_arr = cv2.imencode('.jpg', img)
    return img_arr.tobytes()

root_path = './preprocessing/data'
label_file = f'{root_path}/license-plate.csv'

env_train = lmdb.open('./OCR/data/train')
env_valid = lmdb.open('./OCR/data/valid')

txn_train = env_train.begin(write=True)
txn_valid = env_valid.begin(write=True)

with open(label_file, 'r') as file:
    row_count = sum(1 for row in csv.reader(file)) - 1 # minus 1 because of head line
    num_of_validset = row_count // 10
    num_of_trainset = row_count - num_of_validset

with open(label_file, 'r') as file:

    print(row_count)
    print(num_of_trainset)
    print(num_of_validset)
    csvreader = csv.reader(file)
    for idx, row in enumerate(csvreader):
        if idx == 0:
            # skip first line
            continue
        file_name = row[0]
        label = row[1]
        print(row)
        img = cv2.imread(f'{root_path}/license-plate/{file_name}')
        if idx <= num_of_trainset:
            txn_train.put(b'image-%09d' % idx, img2byte(img))
            txn_train.put(b'label-%09d' % idx, str(label).encode())
        else:
            txn_valid.put(b'image-%09d' % idx, img2byte(img))
            txn_valid.put(b'label-%09d' % idx, str(label).encode())
    
    txn_train.put(b'num-samples', str(num_of_trainset).encode())
    txn_valid.put(b'num-samples', str(num_of_validset).encode())

txn_train.commit()
txn_valid.commit()
env_train.close()
env_valid.close()