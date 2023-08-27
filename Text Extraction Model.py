import os
import easyocr
import numpy as np
import cv2
from PIL import Image
from textblob import TextBlob

def normalize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histeq = cv2.equalizeHist(gray)
    
    gamma = 1.5
    gamma_corr = np.array(255 * (histeq / 255) ** gamma, dtype='uint8')
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gamma_corr)
    return cl

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    smooth = cv2.medianBlur(opening, 3)
    return smooth
image_path = 'C:\\Users\\HP\\Desktop\\data\\img\\63192.png'
reader = easyocr.Reader(['en'])
img = cv2.imread(image_path)
image = normalize_image(img)
image1 = preprocess_image(img)
result = reader.readtext(image)
text = [line[1] for line in result]
clean_text = []
for line in text:
    line = line.replace('[', '').replace(']', '')
    line = ''.join([char for char in line if not char.isdigit()])
    clean_text.append(line)

# Spell Corrector
text_str = ' '.join(text)
blob = TextBlob(text_str)
# Correcting spelling errors
ctext = str(blob.correct())
print(ctext)