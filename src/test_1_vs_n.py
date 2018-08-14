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

def main(args):

    images, boxes = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)  # get images

    # load dataset embedings
    emb_data_file = open(args.embfile, 'r')
    dataset_embs = pickle.load(emb_data_file)
    emb_data_file.close()

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings, 128 dimensions vector
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)

            # same people or totally different
            for retrival_face_emb in emb:
                min_dist = 1000
                for face_in_data_set in dataset_embs:
                    dist = np.sqrt(np.sum(np.square(np.subtract(retrival_face_emb, face_in_data_set))))
                    if min_dist > dist:
                        min_dist = dist
                if min_dist > 1.1:
                    print(min_dist, ' ', 0)
                else:
                    print(min_dist, ' ', 1)

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    tmp_image_paths = []
    with open(image_paths) as f:
        tmp_image_paths = f.read().splitlines()

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    img_list = []
    boxes = []
    i = 0
    flag = True
    for image in tmp_image_paths:  # for each image
        img = cv2.imread(image)  # read images

        if img is None:
            continue

        img_size = np.asarray(img.shape)[0:2]  # get shape: rows, cols
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)  # detect face

        if flag:
            print(type(bounding_boxes))
            flag = False

        if len(bounding_boxes) < 1:
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin/2, 0)  # x1
        bb[1] = np.maximum(det[1] - margin/2, 0)  # y1
        bb[2] = np.minimum(det[2] + margin/2, img_size[1])  # x2
        bb[3] = np.minimum(det[3] + margin/2, img_size[0])  # y2

        boxes.append(bb)

        # crop face and resize to formal size
        cropped = img[bb[1]:bb[3], bb[0]:bb[2],:]
        aligned = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        prewhitened = facenet.prewhiten(aligned)  # pre-process
        img_list.append(prewhitened)

    images = np.stack(img_list)
    return images, boxes

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
        default='./20170511-185253')
    #parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_files', type=str, help='Images to compare', default='./dataset.txt')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--embfile', type=str, help='dataset embeding file', default='./emb.dat')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
