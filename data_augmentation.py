from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import random
import pickle
import os

# load data
file_list = []
class_list = []

DATADIR = "data"
IMG_SIZE = 500
# All the categories neural network will detect
CATEGORIES = ["Adam_Levine"]

# Checking or all images in the data folder
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()
random.shuffle(training_data)

X = []  # features
y = []  # labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# shift = 0.2
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
# Creating the files containing all the information about your model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Opening the files about data
X_train = pickle.load(open("X.pickle", "rb"))
y_train = pickle.load(open("y.pickle", "rb"))

# normalizing data (a pixel goes from 0 to 255)
X = X / 255.0
X_train = X_train.reshape((X_train.shape[0], IMG_SIZE, IMG_SIZE, 3))
X_train = X_train.astype('float32')
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
X_train = X_train[..., ::-1]
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='data/Adam_Levine',
                                     save_prefix='Trang',
                                     save_format='jpg'):
    break

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='data/Adam_Levine',
                                     save_prefix='Trang',
                                     save_format='jpg'):
    break

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='data/Adam_Levine',
                                     save_prefix='Trang',
                                     save_format='jpg'):
    break
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='data/Adam_Levine',
                                     save_prefix='Trang',
                                     save_format='jpg'):
    break

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='data/Adam_Levine',
                                     save_prefix='Trang',
                                     save_format='jpg'):
    break
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9, save_to_dir='data/Adam_Levine',
                                     save_prefix='Trang',
                                     save_format='jpg'):
    break
