import glob
import tensorflow as tf


class DataReader(object):
    def read_file(self, image_dir):
        # Get image & Convert to tensor
        full_image_dirs = sorted(glob.glob(image_dir))
        return full_image_dirs, len(full_image_dirs)

    def read_data(self, full_image_dirs, database, low):
        # Read & Decode image
        image_raw = tf.read_file(full_image_dirs)
        if database == 'espl':
            highest = 8
            image = tf.to_float(tf.image.decode_jpeg(image_raw, channels=3))
        elif database == 'kodak':
            highest = 8
            image = tf.to_float(tf.image.decode_png(image_raw, channels=3, dtype=tf.uint8))
        else:
            highest = 16
            image = tf.to_float(tf.image.decode_png(image_raw, channels=3, dtype=tf.uint16))

        # Quantization
        label = image / (2**highest-1)
        cal = 2 ** (highest - low)
        image = ((image // cal) * cal) / (2**highest-1)
        image = tf.concat([image, tf.ones((tf.shape(image)[0], tf.shape(image)[1], 1)) * low], axis=2)

        # Expand dimension of image and label as 4D tensor
        image, label = tf.expand_dims(image, 0), tf.expand_dims(label, 0)

        return image, label
