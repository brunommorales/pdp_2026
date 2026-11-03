import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
import cv2
import matplotlib
matplotlib.use('agg')
from pylab import array, arange, uint8


def _increase_contrast(image):
    copy = image.copy()

    maxIntensity = 255.0
    x = arange(maxIntensity)

    phi = 1.3
    theta = 1.5
    y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

    copy = (maxIntensity/phi)*(copy/(maxIntensity/theta))**2
    copy = array(copy, dtype=uint8)

    return copy


def _find_contours(image):
    processed = _increase_contrast(image)

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 100:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            return (center, radius)


def _get_filename(file_path):
    return file_path.split("/")[-1]


def _resize_and_center_fundus(image, diameter):
    copy = image.copy()

    contours = _find_contours(image)

    if contours is None:
        return None

    center, radius = contours

    x_min = max(0, int(center[0] - radius))
    y_min = max(0, int(center[1] - radius))
    z = int(radius*2)
    x_max = x_min + z
    y_max = y_min + z

    copy = copy[y_min:y_max, x_min:x_max]

    fx = fy = (diameter / 2) / radius
    copy = cv2.resize(copy, (0, 0), fx=fx, fy=fy)

    shape = copy.shape

    top = bottom = int((diameter - shape[0])/2)
    left = right = int((diameter - shape[1])/2)

    if shape[0] + top + bottom == diameter - 1:
        top += 1

    if shape[1] + left + right == diameter - 1:
        left += 1

    border = [top, bottom, left, right]

    copy = cv2.copyMakeBorder(copy, *border,
                              borderType=cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])
    return copy


def _get_image_paths(images_path):
    return [os.path.join(images_path, fn) for fn in os.listdir(images_path)]


def _resize_and_center_fundus_all(image_paths, save_path, diameter, verbosity):
    num_images = len(image_paths)
    success = 0

    for i, image_path in enumerate(image_paths):
        if verbosity > 0:
            msg = "\r- Preprocessing image: {0:>6} / {1}".format(
                    i+1, num_images)

            sys.stdout.write(msg)
            sys.stdout.flush()

        try:
            image = cv2.imread(os.path.abspath(image_path), -1)

            processed = _resize_and_center_fundus(image, diameter=diameter)

            if processed is None:
                print("Could not preprocess {}...".format(image_path))
            else:
                image_filename = _get_filename(image_path)
                image_jpeg_filename = "{0}.jpg".format(os.path.splitext(
                                        os.path.basename(image_filename))[0])
                output_path = os.path.join(save_path, image_jpeg_filename)

                cv2.imwrite(output_path, processed,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                success += 1
        except AttributeError as e:
            print(e)
            print("Could not preprocess {}...".format(image_path))

    return success


def resize_and_center_fundus(save_path=None, images_path=None, image_paths=None,
                    image_path=None, diameter=299, verbosity=1):
    if save_path is None:
        raise ValueError("Save path not specified!")

    save_path = os.path.abspath(save_path)

    if image_paths is not None:
        return _resize_and_center_fundus_all(image_paths=image_paths,
                                             save_path=save_path,
                                             diameter=diameter,
                                             verbosity=verbosity)

    elif images_path is not None:
        image_paths = _get_image_paths(images_path)
        return _resize_and_center_fundus_all(image_paths=image_paths,
                                             save_path=save_path,
                                             diameter=diameter,
                                             verbosity=verbosity)

    elif image_path is not None:
        return _resize_and_center_fundus_all(image_paths=[image_path],
                                             save_path=save_path,
                                             diameter=diameter,
                                             verbosity=verbosity)


def resize(images_paths, size=299):
    for image_path in images_paths:
        image = cv2.imread(image_path)

        image = cv2.resize(image, (size, size))

        cv2.imwrite(image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def rescale_min_1_to_1(image):
    image = tf.cast(image, tf.float32)
    image = tf.multiply(image, 1. / 127.5)
    return tf.subtract(image, 1.0)


def rescale_0_to_1(image):
    return tf.image.convert_image_dtype(image, tf.float32)
