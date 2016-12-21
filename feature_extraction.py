#!/usr/bin/env python

import os
import glob

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np

#
# Extract deep features from images, based on the tensorflow example at:
#
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
#

# 
# Before running this script, download the tensorflow Inception v3 model to ./imagenet by running:
#
# python .../tensorflow/lib/python3.5/site-packages/tensorflow/models/image/imagenet/classify_image.py --model_dir imagenet
#

#
# The retina photographs should be in ./train
#

model_dir = './imagenet'
images_dir = './train/'
image_list = glob.glob(images_dir + '*')

print('Found %d images' % len(image_list))

def create_graph():
    '''Creates a graph from saved GraphDef file in ./imagenet'''
    
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

        
def extract_features(image_list):
    '''Extract deep features from images in image_list.'''
    
    nb_features = 2048
    features = np.empty((len(image_list), nb_features))

    create_graph()

    with tf.Session() as sess:
        bottleneck_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        print('Processing %i images' % len(image_list))
        for idx, image in enumerate(image_list):
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(bottleneck_tensor, {'DecodeJpeg/contents:0': image_data})
            features[idx, :] = np.squeeze(predictions)

    return features


# Extract features
features = extract_features(image_list)

# Create array of filenames without .jpeg to join with ground-truth classification
keys = [os.path.splitext(os.path.basename(image))[0] for image in image_list]

# Save to disk
np.save('features', features)
np.save('keys', keys)