from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import random
import pickle
import os

desire = 800
CATEGORIES = "spurious copper"
DATADIR = "data"
IMG_SIZE = 100
directory = DATADIR + '/' + CATEGORIES

file_list = []
class_list = []
path = os.path.join(DATADIR, CATEGORIES)
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
training_data = []
path = os.path.join(DATADIR, CATEGORIES)
class_num = CATEGORIES.index(CATEGORIES)
for img in os.listdir(path):
    try:
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
    except Exception as e:
        pass
random.shuffle(training_data)
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
datagen = ImageDataGenerator(rotation_range=90,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
X_train = pickle.load(open("X.pickle", "rb"))
y_train = pickle.load(open("y.pickle", "rb"))
X = X / 255.0
X_train = X_train.reshape((X_train.shape[0], IMG_SIZE, IMG_SIZE, 3))
X_train = X_train.astype('float32')
datagen.fit(X_train)
X_train = X_train[..., ::-1]
while len(os.listdir(directory)) < desire:
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir=directory,
                                         save_prefix='Trang', save_format='jpg'):
        break
