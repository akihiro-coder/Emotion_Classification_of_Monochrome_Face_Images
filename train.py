from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import cv2 as cv
import glob
import os
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import random
from keras.preprocessing.image import ImageDataGenerator



# data augumentation
train_data_dir = './data/train'
test_data_dir = './data/test'






# load train images and convert to numpy array
data_dir = './data/train'
search_pattern = '*.jpg'
images = []
for image_path in glob.glob(os.path.join(data_dir, search_pattern)):
    image = cv.imread(image_path)
    image_expanded = np.expand_dims(image, axis=0)
    images.append(image_expanded)
train_images = np.concatenate(images, axis=0)  # train data (312, 120, 128, 3)


# train data labels
train = pd.read_csv('./data/train_master.tsv', sep='\t')
train_labels_list = train['expression'].tolist()
train_labels = []
for expression in train_labels_list:
    if expression == 'sad':
        expression = 0
    elif expression == 'angry':
        expression = 1
    elif expression == 'neutral':
        expression = 2
    else:
        expression = 3
    train_labels.append(expression)
train_labels = np.array(train_labels)


# make train and validation data
random.seed(0)
random.shuffle(train_images)
train_data_ratio = 0.8
train_images_num = len(train_images)
split_idx = int(train_images_num * train_data_ratio)
x_train, y_train = train_images[:split_idx], train_labels[:split_idx]
x_val, y_val = train_images[split_idx:], train_labels[split_idx:]



# define model
epoch = 200
batch_size = 32
lr = 1e-3
inputs = keras.Input(shape=(120, 128, 3))
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.Flatten()(x)
outputs = layers.Dense(4, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# compile
model.compile(optimizer=Adam(lr=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train
result = model.fit(train_images, train_labels, epochs=epoch, batch_size=batch_size, validation_data=(x_val, y_val))

# visualization
plt.figure(figsize=(15, 12))
#plt.plot(range(1, epoch + 1), result.history['loss'], label='train_loss')
plt.plot(range(1, epoch + 1), result.history['val_loss'], label='val_loss')
#plt.plot(range(1, epoch + 1), result.history['accuracy'], label='train_accuracy')
plt.plot(range(1, epoch + 1), result.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(f"./logs/result_batch{batch_size}_lr{lr}.png")


# save model
model.save(f'./model/my_model_batch{batch_size}_lr{lr}')
