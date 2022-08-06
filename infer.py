from tensorflow import keras
import pandas as pd
import glob
import os
import cv2 as cv
import numpy as np
import csv


test = pd.read_csv('./sample_submit.csv')

# make train images to numpy array
data_dir = './data/test'
search_pattern = '*.jpg'
images = []
for image_path in glob.glob(os.path.join(data_dir, search_pattern)):
    image = cv.imread(image_path)
    image_expanded = np.expand_dims(image, axis=0)
    images.append(image_expanded)
test_images = np.concatenate(images, axis=0)  # train data (312, 120, 128, 3)


# load model
#reconstructed_model = keras.models.load_model('my_model')
reconstructed_model = keras.models.load_model('./my_model_batch32_lr0.001')

# inference
predict_results = reconstructed_model.predict(test_images)  # (312, 4)


# get argmax element
argmax_results = []
for result in predict_results:
    argmax_results.append(np.argmax(result))

submit_results = []
for result in argmax_results:
    if result == 0:
        result = 'sad'
    elif result == 1:
        result = 'angry'
    elif result == 2:
        result = 'neutral'
    else:
        result = 'happy'
    submit_results.append(result)


result_dict = {}
for i in range(312):
    result_dict['test_' + f'{i:04}' + '.jpg'] = submit_results[i]

#  make submit.csv file
date = '2022-08-04'
num = '1'
with open(f'./logs/submit_{date}_{num}.csv', 'w') as f:
    writer = csv.writer(f)
    for i in range(312):
        writer.writerow([list(result_dict.items())[i][0], list(result_dict.items())[i][1]])
