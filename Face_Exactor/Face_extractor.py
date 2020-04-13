# Face detection
from os import listdir
from os.path import isfile, join
import cv2
import numpy

IM_SIZE = 500
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


# extract a single face from a given photograph
def extract_face():
    mypath = 'data'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    images = numpy.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(mypath, onlyfiles[n]), 1)
        # images[n] = images[n][..., ::-1]
        # images[n] = cv2.resize(images[n], (IM_SIZE, IM_SIZE))
        gray = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 5)
        for (x, y, w, h) in faces:
            if y > h//5:
                y_new = y - h//5
            else:
                y_new = y
            if x > w//5:
                x_new = x - w//5
            else:
                x_new = x
            new = images[n][y_new:(y + h + h // 5), x_new:(x + w + w // 5)]
            new = cv2.resize(new, (IM_SIZE, IM_SIZE))
            # images[n] = cv2.rectangle(images[n], (x, y), (x + w, y + h + 30), (255, 0, 0), 2)
            # cv2.imshow('img', new)
            # cv2.waitKey(0)
            cv2.imwrite("image%04i.jpg" % n, new)


extract_face()
