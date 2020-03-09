import cv2
import os
import numpy as np
import pandas as pd
from collections import defaultdict

data_in_path = os.path.join(os.getcwd(), 'gtsrb-german-traffic-sign')
data_out_path = os.path.join(os.getcwd(), 'cleaned-dataset')

category = [0, 1, 4, 6, 8, 9, 10, 12, 13, 15, 16, 17, 18, 22, 25, 27, 29, 32, 35, 36, 37, 38, 39, 40]

def clean_and_create_project_training_dataset():
    for i in range(len(category)):
        cat = category[i]

        img_in_path = os.path.join(data_in_path, 'Train', str(cat))
        img_out_path = os.path.join(data_out_path, 'Train', str(i))
        if not os.path.exists(img_out_path):
            os.makedirs(img_out_path, exist_ok=True)

        for item in os.listdir(img_in_path):
            if str(item[-4:]) == '.png':
                img = cv2.imread(os.path.join(img_in_path, item))
                img = cv2.resize(img, (48, 48))
                cv2.imwrite(os.path.join(img_out_path, item), img)
        
        print("Finished...", img_in_path)

def clean_and_create_project_test_dataset():
    test_label = pd.read_csv(os.path.join(data_in_path, 'Test.csv'))
    test_dict = defaultdict(list)
    for index, entry in test_label.iterrows():
        if entry['ClassId'] in test_dict.keys():
            test_dict[entry['ClassId']].append(entry['Path'][5:])
        else:
            test_dict[entry['ClassId']] = [entry['Path'][5:]]

    img_in_path = os.path.join(data_in_path, 'Test')
    for idx, cat in enumerate(category):
        img_out_path = os.path.join(data_out_path, 'Test', str(idx))
        if not os.path.exists(img_out_path):
            os.makedirs(img_out_path, exist_ok=True)

        imgs = test_dict.get(cat)
        for item in imgs:
            img = cv2.imread(os.path.join(img_in_path, item))
            img = cv2.resize(img, (48, 48))
            cv2.imwrite(os.path.join(img_out_path, item), img)
        print("Finished...", cat)

if __name__ == "__main__":
    if not os.path.exists(data_out_path):
        os.makedirs(data_out_path, exist_ok=True)

    # clean_and_create_project_training_dataset()
    clean_and_create_project_test_dataset()