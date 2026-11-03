import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import argparse
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='This script checks whether the tfrecord files were generated correctly.')
parser.add_argument("-t", "--tfrec_dir", help="Directory where the tfrecord files reside.", default="./tfrec")
parser.add_argument("-b", "--batch", help="Batch size to load (default=256).", default = 256)
parser.add_argument("-i", "--img_pixel", help="Diameter of the images (in pixels) (default=299).", default= 299)

args = parser.parse_args()
TFREC_DIR = str(args.tfrec_dir)
BATCH_SIZE = args.batch
IMG_PIXEL = args.img_pixel

IMAGE_SIZE= [IMG_PIXEL,IMG_PIXEL]
AUTO = tf.data.experimental.AUTOTUNE

def has_tfrecord_files(directory):
    return any(file.endswith(".tfrec") for file in os.listdir(directory))


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "imagem": tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'retinopatia' : tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['imagem'])
    label = tf.cast(example['retinopatia'], tf.int32)
    name = example['image_name']


    return image, label, name

def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord)
    return dataset

def get_training_dataset(filenames):
    dataset = load_dataset(filenames, labeled=True)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def show_batch_v2(image_batch, label_batch, name, dataset):
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for n, ax in enumerate(axes.flat):
        ax.imshow(image_batch[n] / 255.0)
        ax.set_title(label_batch[n])
        ax.axis("off")

    plt.savefig(dataset + ".png", format="png", bbox_inches="tight")
    plt.close()

if not os.path.exists(TFREC_DIR):
    print(f"The directory {TFREC_DIR} does not exist.")
    sys.exit()

if not has_tfrecord_files(TFREC_DIR):
    print("The directory does not contain TFRecord files.")
    sys.exit()

train_filenames = tf.io.gfile.glob(os.path.join(TFREC_DIR,'train*.tfrec'))
valid_filenames = tf.io.gfile.glob(os.path.join(TFREC_DIR,'test*.tfrec'))
print('Number of images in train tfrecords: ', count_data_items(train_filenames))
print('Number of images in test tfrecords: ', count_data_items(valid_filenames))

filenames = [train_filenames, valid_filenames]

dataset = "train"
for files in filenames:

    data = get_training_dataset(files)

    image_batch, label_batch, name = next(iter(data))

    show_batch_v2(image_batch.numpy(), label_batch.numpy(), name.numpy(), dataset)
    dataset = "test"
