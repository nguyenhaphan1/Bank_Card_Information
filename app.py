from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os
import model
import cv2
import tensorflow as tf
import numpy as np
app = Flask(__name__)
upload_folder = os.path.join("static", "uploads")
app.config['UPLOAD'] = upload_folder

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('homePage3.html', tensor_input='')

# @app.route('/upload', methods=['POST'])
# def preview():
#     # Main page
#     if request.method == 'POST':
#         f = request.files['file']
#         path = os.path.join(app.config['UPLOAD'], secure_filename(f.filename))
#         f.save(path)
#     return render_template('homePage2.html', preview=path)

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        path = os.path.join(app.config['UPLOAD'], secure_filename(f.filename))
        f.save(path)
        img = cv2.imread(path)
        number, predicted_image = model.detect_img(img)
        cv2.imwrite(os.path.join(upload_folder, "image.jpg"), predicted_image)
        path = os.path.join(upload_folder, "image.jpg")
    return render_template('homePage3.html', title = number, image=path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)

