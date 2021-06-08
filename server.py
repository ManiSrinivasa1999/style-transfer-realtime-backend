from itertools import product
import os
import cv2
import base64

from flask import Flask
from flask_cors import CORS
from flask import request

import numpy as np
import tensorflow as tf

from adain.image import load_image, prepare_image, load_mask, save_image
from adain.coral import coral
from adain.nn import build_vgg, build_decoder
from adain.norm import adain
from adain.weights import open_weights
from adain.util import get_filename, get_params, extract_image_names_recursive

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # No GPU so -1
data_format = 'channels_last' # w, h ,c
vgg_weights = 'models/vgg19_weights_normalized.h5' #  from transfer learning 
decoder_weights = 'models/decoder_weights.h5' # network structure
alpha = 1.0 # style intensity
config = tf.ConfigProto(
    device_count = {'GPU': 0}
)


def _build_graph(vgg_weights, decoder_weights, alpha, data_format):
    if data_format == 'channels_first':
        image = tf.placeholder(shape=(None, 3, None, None), dtype=tf.float32)
        content = tf.placeholder(shape=(1, 512, None, None), dtype=tf.float32)
        style = tf.placeholder(shape=(1, 512, None, None), dtype=tf.float32)
    else:
        image = tf.placeholder(shape=(None, None, None, 3), dtype=tf.float32)
        content = tf.placeholder(shape=(1, None, None, 512), dtype=tf.float32)
        style = tf.placeholder(shape=(1, None, None, 512), dtype=tf.float32)

    target = adain(content, style, data_format=data_format)
    weighted_target = target * alpha + (1 - alpha) * content

    with open_weights(vgg_weights) as w:
        vgg = build_vgg(image, w, data_format=data_format)
        encoder = vgg['conv4_1']

    if decoder_weights:
        with open_weights(decoder_weights) as w:
            decoder = build_decoder(weighted_target, w, trainable=False,
                                    data_format=data_format)
    else:
        decoder = build_decoder(weighted_target, None, trainable=False,
                                data_format=data_format)

    return image, content, style, target, encoder, decoder


style_path = 'input/style/woman_with_hat_matisse.jpg'
style_name = get_filename(style_path)
style_image = load_image(style_path, 0, None)
# style image should be numpy array
style_image = prepare_image(style_image, True, data_format)

app = Flask(__name__)
CORS(app)

image, content, style, target, encoder, decoder = _build_graph(vgg_weights,
                                                               decoder_weights,
                                                               alpha,
                                                               data_format)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
style_feature = sess.run(encoder, feed_dict={
    image: style_image[np.newaxis, :]
})

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/detect-image/', methods=['POST'])
def predict():
    global image, content, style, target, encoder, decoder, style_feature
    img = request.get_json()['image']
    content_image = data_uri_to_cv2_img(img)

    content_image = prepare_image(content_image, True, data_format)
    content_feature = sess.run(encoder, feed_dict={
        image: content_image[np.newaxis, :]
    })
    target_feature = sess.run(target, feed_dict={
        content: content_feature,
        style: style_feature
    })
    output = sess.run(decoder, feed_dict={
        content: content_feature,
        target: target_feature
    })
    if data_format == 'channels_first':
        result = np.transpose(output[0], [1, 2, 0])  # CHW --> HWC
    else:
        result = np.transpose(output[0], [0, 1, 2])
    result *= 255
    result = np.clip(result, 0, 255)
    _, buffer = cv2.imencode('.png', result)
    return 'data:image/png;base64,' + base64.b64encode(buffer).decode('utf-8')


app.run(host='0.0.0.0', port=8085)
