from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import time
import imp
import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import keras.models
from keras.models import load_model
from keras.preprocessing import image
from keras import backend
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify, make_response, Response
from gevent.pywsgi import WSGIServer

import innvestigate.utils as iutils
import innvestigate.applications.imagenet
import innvestigate
import keras

import os
import warnings
# Importing Image module from PIL package
from PIL import Image
import PIL
from functools import wraps
import urllib
import urllib.request
from flask_cors import CORS
import matplotlib  # nopep8
matplotlib.use('agg')  # nopep8
import matplotlib.pyplot as plt  # nopep8
import cv2
import base64

# global vars for easy reusability
global model
# initialize these variables


def get_model():
    model = keras.models.load_model(
        "/root/py-deploy/api/deployedNewModel.h5")
    model._make_predict_function()

    model_wo_softmax = iutils.keras.graph.model_wo_softmax(keras.models.load_model(
        "/root/py-deploy/api/deployedNewModel.h5"))
    model_wo_softmax._make_predict_function()
    global graph
    graph = tf.get_default_graph()
    return [model, model_wo_softmax]


model = get_model()

dir_path = os.path.dirname(os.path.realpath(__file__))
# Use utility libraries to focus on relevant iNNvestigate routines.
eutils = imp.load_source(
    "utils", "/root/py-deploy/api/utils/utils.py")
imgnetutils = imp.load_source(
    "utils_imagenet", "/root/py-deploy/api/utils/utils_imagenet.py")

# Load the model definition.
tmp = getattr(innvestigate.applications.imagenet,
              os.environ.get("NETWORKNAME", "vgg16"))
net = tmp(load_weights=True, load_patterns="relu")


def runNonImage():
    input_range = net["input_range"]
    # Handle input depending on model and backend.
    images, label_to_class_name = eutils.get_imagenet_data(
        net["image_shape"][0])
    input_range = net["input_range"]
    noise_scale = (input_range[1]-input_range[0]) * 0.1
    text = []
    with graph.as_default():
        # Create analyzers.
        for i, (x, y) in enumerate(images):
            x = x[None, :, :, :]
            x_pp = imgnetutils.preprocess(x, net)
            # Predict final activations, probabilites, and label.
            prob = model[0].predict_on_batch(x_pp)[0]
            y_hat = prob.argmax()
            # Save prediction info:
            # Save prediction info:
            text.append(("%s" % label_to_class_name[y],    # ground truth label
                         "%.2f" % prob.max(),              # probabilistic softmax output
                         "%s" % label_to_class_name[y_hat]  # predicted label
                         ))
    print("rext", text)
    return text


def runUsingModel(analysisType):
    selectedAnalysis = "DeepTaylor" if analysisType == None else str(
        analysisType)
    print('selectedAnalysis', selectedAnalysis)
    patterns = net["patterns"]
    input_range = net["input_range"]
    # Handle input depending on model and backend.
    channels_first = keras.backend.image_data_format() == "channels_first"
    color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None
    images, label_to_class_name = eutils.get_imagenet_data(
        net["image_shape"][0])
    input_range = net["input_range"]
    noise_scale = (input_range[1]-input_range[0]) * 0.1
    ANALYSIS_OPTIONS = {
        "Input": ("input",                 {},
                  imgnetutils.image,         "Input"),

        # Function
        "Gradient": ("gradient",              {"postprocess": "abs"},
                     imgnetutils.graymap,       "Gradient"),
        "SmoothGrad": ("smoothgrad",            {"augment_by_n": 64,
                                                 "noise_scale": noise_scale,
                                                 "postprocess": "square"}, imgnetutils.graymap,       "SmoothGrad"),

        # Signal
        "Deconvnet": ("deconvnet",             {},
                      imgnetutils.bk_proj,       "Deconvnet"),
        "Guided-Backprop":  ("guided_backprop",       {},
                             imgnetutils.bk_proj,       "Guided Backprop",),
        "PatternNet": ("pattern.net",           {"patterns": patterns},
                       imgnetutils.bk_proj,       "PatternNet"),

        # Interaction
        "PatternAttribution": ("pattern.attribution",   {"patterns": patterns},
                               imgnetutils.heatmap,       "PatternAttribution"),
        "DeepTaylor": ("deep_taylor.bounded",   {"low": input_range[0],
                                                 "high": input_range[1]}, imgnetutils.heatmap,       "DeepTaylor"),
        "Input*Gradient": ("input_t_gradient",      {},
                           imgnetutils.heatmap,       "Input * Gradient"),
        "Integrated-Gradients": ("integrated_gradients",  {
            "reference_inputs": input_range[0], "steps": 64}, imgnetutils.heatmap,       "Integrated Gradients"),
        "LRP-Z": ("lrp.z",                 {},
                  imgnetutils.heatmap,       "LRP-Z"),
        "LRP-Epsilon": ("lrp.epsilon",           {"epsilon": 1},
                        imgnetutils.heatmap,       "LRP-Epsilon"),
        "LRP-PresetAFlat": ("lrp.sequential_preset_a_flat", {"epsilon": 1},
                            imgnetutils.heatmap,       "LRP-PresetAFlat"),
        "LRP-PresetBFlat": ("lrp.sequential_preset_b_flat", {"epsilon": 1},
                            imgnetutils.heatmap,       "LRP-PresetBFlat"),
        "All": ("lrp.asdfasdf", {"sadf": 1},
                imgnetutils.heatmap,       "LRP-dd")
    }
    methods = [ANALYSIS_OPTIONS["Input"],
               ANALYSIS_OPTIONS[selectedAnalysis]]
    print("selectedAnalysis", selectedAnalysis)
    if(selectedAnalysis == "All"):
        methods = [ANALYSIS_OPTIONS["Input"], ANALYSIS_OPTIONS["Gradient"], ANALYSIS_OPTIONS["SmoothGrad"], ANALYSIS_OPTIONS["Deconvnet"],
                   ANALYSIS_OPTIONS["PatternNet"], ANALYSIS_OPTIONS["PatternAttribution"], ANALYSIS_OPTIONS["DeepTaylor"], ANALYSIS_OPTIONS[
                       "Input*Gradient"], ANALYSIS_OPTIONS["Integrated-Gradients"], ANALYSIS_OPTIONS["LRP-Z"], ANALYSIS_OPTIONS["LRP-Epsilon"],
                   ANALYSIS_OPTIONS["LRP-PresetAFlat"], ANALYSIS_OPTIONS["LRP-PresetBFlat"]]

    with graph.as_default():
        print("starting the analysis")

        # Create analyzers.
        analyzers = []
        for method in methods:
            try:
                analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                        # model without softmax output
                                                        model[1],
                                                        **method[1])      # optional analysis parameters
            except innvestigate.NotAnalyzeableModelException:
                # Not all methods work with all models.
                analyzer = None
            analyzers.append(analyzer)

        analysis = np.zeros(
            [len(images), len(analyzers)]+net["image_shape"]+[3])

        # Create analyzers.
        text = []
        for i, (x, y) in enumerate(images):
            x = x[None, :, :, :]
            x_pp = imgnetutils.preprocess(x, net)
            # Predict final activations, probabilites, and label.
            prob = model[0].predict_on_batch(x_pp)[0]
            y_hat = prob.argmax()
            # Save prediction info:
            # Save prediction info:
            text.append(("%s" % label_to_class_name[y],    # ground truth label
                         "%.2f" % prob.max(),              # probabilistic softmax output
                         "%s" % label_to_class_name[y_hat]  # predicted label
                         ))
            print("text", text)
            for aidx, analyzer in enumerate(analyzers):
                if methods[aidx][0] == "input":
                    # Do not analyze, but keep not preprocessed input.
                    a = x/255
                elif analyzer:
                    # Analyze.
                    a = analyzer.analyze(x_pp)
                    # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                    a = imgnetutils.postprocess(
                        a, color_conversion, channels_first)
                    # Apply analysis postprocessing, e.g., creating a heatmap.
                    print("applyng analysis")
                    a = methods[aidx][2](a)
                # Store the analysis.
                analysis[i, aidx] = a[0]
                # Prepare the grid as rectengular list
        grid = [[analysis[i, j] for j in range(analysis.shape[1])]
                for i in range(analysis.shape[0])]

        label, prob, pred = zip(*text)
        row_labels_left = [('pred: {}'.format(pred[i]))
                           for i in range(len(label))]
        row_labels_right = [('prob: {}'.format(prob[i]))
                            for i in range(len(label))]
        col_labels = [''.join(method[3]) for method in methods]
        # Plot the analysis.
        eutils.plot_image_grid(grid, row_labels_left,
                               row_labels_right, col_labels, "masterimage")

    return text


def getImage(url):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent': user_agent, }
    request = urllib.request.Request(url, None, headers)
    response = urllib.request.urlopen(request)
    image = Image.open(response).convert('RGB')
    image.save(
        "/root/py-deploy/api/utils/images/n02799071_986.jpg")
    return True


app = Flask(__name__, static_folder='../build', static_url_path='/')
CORS(app)

# uploads_dir = os.path.join(app.instance_path, 'uploads')
# os.makedirs(uploads_dir, exists_ok=True)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/time')
def get_current_time():
    return {'time': time.time()}


print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/check', methods=['POST'])
def reload():
    q = request.query_string
    u = str(q.decode("utf-8"))
    entireString = u.lstrip('q=')
    analysisIdx = entireString.find('type')
    analysisType = entireString[analysisIdx+5:]

    imgStr = str(entireString)[:analysisIdx-1].replace('"', "")
    # Add photo
    getImage(imgStr)
    preds = runUsingModel(analysisType)
    return jsonify(preds)


@app.route('/image', methods=['GET'])
def getImageAnalysis():
    img = cv2.imread('masterimage.png')
    # encode image as jpeg
    string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')
    # build a response dict to send back to client
    dict = {
        'img': string
    }

    return dict


@app.route('/predict', methods=['POST', 'GET', 'PUT'])
def upload():
    if request.method == 'POST':
        data = request.form['url']
        # # Add photo
        print('~~~~~~~~>>>', data)
        getImage(data)
        out = {'preds': 22}
        return jsonify(out)
    if request.method == 'GET':
        # Make prediction
        preds = runNonImage()
        out = preds
        return jsonify(out)
    return None


@app.errorhandler(404)
def not_found_error(error):
    print("error", error)
    app.send_static_file('index.html')


@app.errorhandler(500)
def internal_error(error):
    print("error", error)
    app.send_static_file('index.html')


app.run(host='0.0.0.0', port=5000, debug=True)
