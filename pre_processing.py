import cv2
import os
import numpy as np

data_in_path = os.path.join(os.getcwd(), 'gtsrb-german-traffic-sign')
data_out_path = os.path.join(os.getcwd(), 'cleaned-dataset')

# category = [0, 1, 4, 6, 8, 9, 10, 12, 13, 15, 16, 17, 18, 22, 25, 27, 29, 32, 35, 36, 37, 38, 39, 40]
category = [0]

def clean_and_create_project_dataset():
    for i in range(len(category)):
        cat = category[i]

        img_in_path = os.path.join(data_in_path, 'Train', str(cat))
        if not os.path.exists(img_in_path):
            os.makedirs(img_in_path, exist_ok=True)

        img_out_path = os.path.join(data_out_path, 'Train', str(i))
        if not os.path.exists(img_out_path):
            os.makedirs(img_out_path, exist_ok=True)

        for item in os.listdir(img_in_path):
            if str(item[-4:]) == '.png':
                img = cv2.imread(os.path.join(img_in_path, item))
                print(os.path.join(img_in_path, item))
                img = cv2.resize(img, (48, 48))
                cv2.imwrite(os.path.join(img_out_path, item))
        
        print("Finished...", img_in_path)

clean_and_create_project_dataset()