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

def main(args):

    images, ori_images, boxes = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)  # get images
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }  # here 2 images
            emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = len(args.image_files)

            num_faces = len(images)
            print(num_faces)
            for i in range(0, num_faces, 2):
                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[i + 1,:]))))

                b_a = boxes[i]
                b_b = boxes[i + 1]
                if dist < 1.10:  # draw green
                    cv2.rectangle(ori_images[i], (int(b_a[0]), int(b_a[1])), (int(b_a[2]), int(b_a[3])), (0, 255, 0), 2)
                    cv2.rectangle(ori_images[i + 1], (int(b_b[0]), int(b_b[1])), (int(b_b[2]), int(b_b[3])), (0, 255, 0), 2)
                    misc.imshow(ori_images[i])
                    misc.imshow(ori_images[i + 1])
                    #cv2.imshow("person a", ori_images[i])
                    #cv2.imshow("person b", ori_images[i + 1])
                    #cv2.waitKey(0)

                    print("same id")
                else:            # draw red
                    cv2.rectangle(ori_images[i], (int(b_a[0]), int(b_a[1])), (int(b_a[2]), int(b_a[3])), (255, 0, 0), 2)
                    cv2.rectangle(ori_images[i + 1], (int(b_b[0]), int(b_b[1])), (int(b_b[2]), int(b_b[3])), (255, 0, 0), 2)
                    misc.imshow(ori_images[i])
                    misc.imshow(ori_images[i + 1])
                    #cv2.imshow("person a", ori_images[i])
                    #cv2.imshow("person b", ori_images[i + 1])
                    #cv2.waitKey(0)
                    print("different id")


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths = image_paths[:]
    img_list = []
    ori_images = []
    boxes = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')

        #misc.imshow(img)

        ori_images.append(img)

        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])

        boxes.append(bb)

        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images, ori_images, boxes

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
        default='./20170511-185253')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
