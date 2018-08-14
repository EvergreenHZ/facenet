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

def main(args):

    images, boxes = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)  # get images

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

            f = 128
            t = AnnoyIndex(f)
            num_faces = len(emb)
            for i in xrange(num_faces):
                t.add_item(i, emb[i])

            t.build(50)
            t.save('facembs')

            print('Finish building')


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    tmp_image_paths = []
    with open(image_paths) as f:
        tmp_image_paths = f.read().splitlines()

    print('length: ' + str(len(tmp_image_paths)))

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
    valid_name_list = []
    for image in tmp_image_paths:  # for each image
        img = cv2.imread(image)  # read images

        if img is None:
            continue

        img_size = np.asarray(img.shape)[0:2]  # get shape: rows, cols
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)  # detect face

        if len(bounding_boxes) < 1:
          continue

        valid_name_list.append(image)
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

    for i in xrange(len(valid_name_list)):
        print(valid_name_list[i])
    with open('valid_name_list', 'wb') as f:
        pickle.dump(valid_name_list, f)
    images = np.stack(img_list)
    print('load and align: DONE')
    return images, boxes

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
        default='./20170511-185253/20170511-185253.pb')
    parser.add_argument('--image_files', type=str, help='Images to compare', default='./hello.txt')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
