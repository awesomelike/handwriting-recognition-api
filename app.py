from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import os
import sys
import cv2
from flask import Flask, jsonify, request, redirect, flash, url_for, send_from_directory
from werkzeug.utils import secure_filename
import string 
import random 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
BASE_URL = ''
try:
  BASE_URL = sys.argv[2]
except:
  BASE_URL = 'http://localhost:3000'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recognize(image_path):
  print("loading CNN model...")

  model_path = 'handwriting.model'
  # image_path = 'image.jpg'

  model = load_model(model_path)
  image = cv2.imread(image_path)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # perform edge detection, find contours in the edge map, and sort the
  # resulting contours from left-to-right
  edged = cv2.Canny(blurred, 30, 150)
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sort_contours(cnts, method="left-to-right")[0]

  # initialize the list of contour bounding boxes and associated
  # characters that we'll be OCR'ing
  chars = []

  # loop over the contours
  for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    # filter out bounding boxes, ensuring they are neither too small
    # nor too large
    if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
      # extract the character and threshold it to make the character
      # appear as *white* (foreground) on a *black* background, then
      # grab the width and height of the thresholded image
      roi = gray[y:y + h, x:x + w]
      thresh = cv2.threshold(roi, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      (tH, tW) = thresh.shape

      # if the width is greater than the height, resize along the
      # width dimension
      if tW > tH:
        thresh = imutils.resize(thresh, width=32)

      # otherwise, resize along the height
      else:
        thresh = imutils.resize(thresh, height=32)

      # re-grab the image dimensions (now that its been resized)
      # and then determine how much we need to pad the width and
      # height such that our image will be 32x32
      (tH, tW) = thresh.shape
      dX = int(max(0, 32 - tW) / 2.0)
      dY = int(max(0, 32 - tH) / 2.0)

      # pad the image and force 32x32 dimensions
      padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0))
      padded = cv2.resize(padded, (32, 32))

      # prepare the padded image for classification via our
      # handwriting OCR model
      padded = padded.astype("float32") / 255.0
      padded = np.expand_dims(padded, axis=-1)

      # update our list of characters that will be OCR'd
      chars.append((padded, (x, y, w, h)))

  # extract the bounding box locations and padded characters
  boxes = [b[1] for b in chars]
  chars = np.array([c[0] for c in chars], dtype="float32")

  # OCR the characters using our handwriting recognition model
  preds = model.predict(chars)

  labels = "0123456789"
  labels += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  labels = [l for l in labels]

  for (pred, (x, y, w, h)) in zip(preds, boxes):
    i = np.argmax(pred)
    prob = pred[i]
    label = labels[i]

    print("(PROBABILITY) {} - {:.2f}%".format(label, prob * 100))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 200), 2)
    cv2.putText(image, label, (x - 10, y - 10),
      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    
  filename = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k = 8)) + '.jpg'
  print('AZAMAT', filename)
  filepath = './output/' + filename
  cv2.imwrite(filepath, image)
  return filename
  
  # show the image
  # cv2.imshow('Result', image)
  # cv2.waitKey(0)

@app.route('/recognize', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        filename = secure_filename(file.filename)
        uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_path)
        output_image = recognize(uploaded_path)
        
        return jsonify({ 'image': BASE_URL + '/' + 'output' + '/' + output_image })

@app.route('/output/<path:path>')
def send_js(path):
    return send_from_directory('output', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

