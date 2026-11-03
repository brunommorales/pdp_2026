import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from os import makedirs
from os.path import exists
from shutil import rmtree
import time
import pandas as pd
import tensorflow as tf
import cv2

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(img, name, label):
    feature = {
            'imagem': _bytes_feature(img),
            'image_name': _string_feature(name),
            'retinopatia': _int64_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_arguments():
    parser = argparse.ArgumentParser(description='This script creates tfrecord files based on already processed and standardized images. See more: python3 create-tfrecord.py -h')

    parser.add_argument('-b', '--base_dir', type=str, default='/hcpa/data', help='Base directory path')
    parser.add_argument('-f1', '--labels_train', type=str, default='/csv/labels_train.csv', help='File labels path for train dataset')
    parser.add_argument('-f2', '--labels_test', type=str, default='/csv/labels_test.csv', help='File labels path for test dataset')
    parser.add_argument('-d', '--data_dir', type=str, default='images', help='Data images directory path for create tfrecords')
    parser.add_argument('-t', '--tfrec_dir', type=str, default='tfrec', help='TFRecord directory path')
    parser.add_argument('--img_pixel', type=int, default=299, help='Image pixel size')
    parser.add_argument('--size', type=int, default=512, help='Number of images in each tfrecord file')

    return parser.parse_args()


args = parse_arguments()
BASE_DIR = args.base_dir
LABELS_TRAIN = os.path.join(BASE_DIR, args.labels_train)
LABELS_TEST = os.path.join(BASE_DIR, args.labels_test)
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, args.data_dir)
TFREC_DIR = os.path.join(BASE_DIR, args.tfrec_dir)
IMG_PIXEL = args.img_pixel
SIZE = args.size


if exists(TFREC_DIR):
    rmtree(TFREC_DIR)
makedirs(TFREC_DIR)

train = pd.read_csv(LABELS_TRAIN)
valid = pd.read_csv(LABELS_TEST)

datasets = [train, valid]
for dataset in datasets:
    print(dataset.groupby(['retinopatia']).size())


tfrec_name = "train"
for dataset in datasets:

    IMGS = dataset['imagem'].values

    CT = len(IMGS)//SIZE + int(len(IMGS)%SIZE!=0)

    count = 0

    for j in range(CT):
        print();
        print('Writing TFRecord %i of %i...'%(j+1, CT))

        tStart = time.time()

        CT2 = min(SIZE, len(IMGS)-j*SIZE)

        with tf.io.TFRecordWriter(os.path.join(TFREC_DIR, f'{tfrec_name}%.2i-%i.tfrec'%(j,CT2))) as writer:
            for k in range(CT2):
                index = SIZE*j+k
                img_path = os.path.join(TRAIN_IMAGES_DIR, dataset.iloc[index].imagem)

                img = cv2.imread(img_path)


                img = cv2.imencode('.png', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tobytes()

                name = dataset.iloc[index].imagem
                row = dataset.iloc[index]

                level = row['retinopatia']

                example = serialize_example(img, name, level)

                writer.write(example)

                if (k + 1) % 10 == 0 or k == CT2 - 1:
                    progress_percentage = ((k + 1) / CT2) * 100
                    print(f'  Writing {progress_percentage:.2f}% complete', end='\r')


        tEnd = time.time()

        print('')
        print('Elapsed: ', round((tEnd - tStart),1), ' (sec)')

    tfrec_name = "test"
