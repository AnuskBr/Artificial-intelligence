# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 17:53:26 2024

@author: Anusk
"""

from PIL import Image
from PIL import ImageDraw
import cv2 as cv
from IPython.display import display
import timeit

def show_drept(faces):
    pil_img=Image.open('poza.jpg').convert("RGB")
    drawing=ImageDraw.Draw(pil_img)
    for x,y,w,h in faces:
        drawing.rectangle((x,y,x+w,y+h), outline="green",width=8)
    display(pil_img)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
img = cv.imread('poza.jpg')

cv_img_bin=cv.threshold(img,120,255,cv.THRESH_BINARY)[1] 
# din lista intoarsa, ne intereseaza a doua valoare

faces = face_cascade.detectMultiScale(cv_img_bin,1.95)
def detect_faces():
    face_cascade.detectMultiScale(cv_img_bin, 1.95)

print(timeit.timeit(detect_faces, number=10))

show_drept(faces)