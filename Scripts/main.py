import cv2

alg = 'models/haarcascade_frontalface_default.xml'

haar_cascade = cv2.CascadeClassifier(alg)

file_name = 'data/raw/faces.jpeg'

img = cv2.imread(file_name, 0)