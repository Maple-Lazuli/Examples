from flask import Flask, request, Response, send_file
from flask_cors import CORS
import time
import json
import hashlib
import pickle
import os

import tensorflow as tf
import numpy as np
from PIL import Image

tf.compat.v1.disable_eager_execution()
from PIL import Image, ImageOps
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)


@app.before_first_request
def run_first():
    global sess
    global softmax_classifier
    global input_image
    global hold_prob
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph("./model/mnist.meta")
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint("./model"))
    graph = tf.compat.v1.get_default_graph()
    input_image = graph.get_tensor_by_name("mnist_model/X:0")
    classifier_label = graph.get_tensor_by_name("mnist_model/Y_Prediction/y_pred:0")
    softmax_classifier = tf.compat.v1.math.softmax(classifier_label)
    hold_prob = graph.get_tensor_by_name("mnist_model/hold_prob:0")


@app.route("/images", methods=["POST"])
def save_image():
    # save and format the image from the client
    file = request.json['data']['imageData']
    im = Image.open(BytesIO(base64.b64decode(file[22:])))
    im = im.resize((28, 28))
    im = ImageOps.grayscale(im)
    # create unique file  name
    prehash = str(time.time())
    file_name = hashlib.sha256(bytes(prehash, 'utf-8')).hexdigest()
    # save the file and return the new name to the user
    im.save(f"images/{file_name}.png")
    return Response(json.dumps({'image': file_name}), status=200, mimetype='application/json')


@app.route("/image", methods=["GET"])
def get_image():
    image_name = request.args.get('image')
    return send_file("images/" + image_name + '.png', mimetype='image/png')


@app.route("/prediction", methods=["GET"])
def get_prediction():
    global sess
    global softmax_classifier
    global input_image
    global hold_prob

    image_png = Image.open("images/" + request.args.get('image') + ".png")
    image_arr = np.array(image_png).reshape(-1, 784)
    y_pred = sess.run(softmax_classifier, feed_dict={input_image: image_arr, hold_prob: 1.0})
    predictions = []
    for idx in range(0, 10):
        predictions.append(float(y_pred[0][idx]))
    return Response(json.dumps({"predictions": predictions}), status=200, mimetype='application/json')


def main():
    app.run(host='0.0.0.0', port=3033)


if __name__ == "__main__":
    main()
