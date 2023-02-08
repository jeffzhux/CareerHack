import csv
import cv2

def img2byte(img):
    is_success, img_arr = cv2.imencode('.jpg', img)
    return img_arr.tobytes()

root_path = './preprocessing/data'
label_file = f'{root_path}/license-plate.csv'


with open(label_file, 'r') as file:
    row_count = sum(1 for row in csv.reader(file)) - 1 # minus 1 because of head line
    num_of_validset = row_count // 10
    num_of_trainset = row_count - num_of_validset

with open(label_file, 'r') as file:

    csvreader = csv.reader(file)
    train_id = 1
    valid_id = 1
    for idx, row in enumerate(csvreader):
        if idx == 0:
            # skip first line
            continue
        file_name = row[0]
        label = row[1]

        img = cv2.imread(f'{root_path}/license-plate/{file_name}')
        if idx <= num_of_trainset:
            print(b'label-%09d' % train_id, str(label).encode(), file_name)
            train_id += 1
            break
        else:
            valid_id += 1
    