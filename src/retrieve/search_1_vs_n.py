"""Performs face alignment and calculates L2 distance between the embeddings of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import cv2
import pickle
from annoy import AnnoyIndex

model_path = './20170511-185253/20170511-185253.pb'
image_size = 160
margin = 44
gpu_memory_fraction = 1.0

u = AnnoyIndex(128)
u.load('facembs')

candidate_image = 'dataset/celea_1120.jpg'

def main(image_path):

    face_list = load_and_align_data(image_path, image_size, margin, gpu_memory_fraction)  # get images

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings, 128 dimensions vector
            feed_dict = { images_placeholder: face_list, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)

            #print(str(emb[0]))
            l = u.get_nns_by_vector(emb[0], 100)

            dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], u.get_item_vector(l[0])))))
            #print(str(dist))
            #print(str(u.get_nns_by_vector(emb[0], 100)))
            #print(str(len(u.get_nns_by_vector(emb[0], 101))))


def load_and_align_data(image_path, image_size, margin, gpu_memory_fraction):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    img = cv2.imread(image_path)  # read images

    if img is None:
        return None

    img_size = np.asarray(img.shape)[0:2]  # get shape: rows, cols
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)  # detect face

    if len(bounding_boxes) < 1:
        return None
    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin/2, 0)  # x1
    bb[1] = np.maximum(det[1] - margin/2, 0)  # y1
    bb[2] = np.minimum(det[2] + margin/2, img_size[1])  # x2
    bb[3] = np.minimum(det[3] + margin/2, img_size[0])  # y2

    # crop face and resize to formal size
    cropped = img[bb[1]:bb[3], bb[0]:bb[2],:]
    aligned = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    prewhitened = facenet.prewhiten(aligned)  # pre-process

    return_list = []
    return_list.append(prewhitened)
    print('detect & align')
    return return_list

main(candidate_image)
