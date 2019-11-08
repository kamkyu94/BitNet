import glob
import tensorflow as tf


class DataReader(object):
    def __init__(self, image_dir, is_training, with_bit_info):
        self.is_training = is_training
        full_image_dirs, self.img_num = self.read_file(image_dir)
        self.full_image_dir = tf.train.slice_input_producer([full_image_dirs], shuffle=is_training)
        self.with_bit_info = with_bit_info

    def read_file(self, image_dir):
        # Get image & Convert to tensor
        full_image_dirs = sorted(glob.glob(image_dir))
        full_image_dirs_tensor = tf.convert_to_tensor(full_image_dirs, dtype=tf.string)

        return full_image_dirs_tensor, len(full_image_dirs)

    def read_data(self, low):
        # Read & Decode image
        image_raw = tf.read_file(self.full_image_dir[0])
        image = tf.to_float(tf.image.decode_png(image_raw, channels=3, dtype=tf.uint16))

        # Training and testing
        if self.is_training:
            image = self.data_augmentation(image)

        # Quantization
        label = image / 65535.
        cal = 2 ** (16 - low)
        image = ((label * 65535. // cal) * cal) / 65535.

        # Concatenate quantzation bit information
        if self.with_bit_info:
            image = tf.concat([image, tf.ones((tf.shape(image)[0], tf.shape(image)[1], 1)) * low], axis=2)

        # Expand dimension of image and label as 4D tensor
        image, label = tf.expand_dims(image, 0), tf.expand_dims(label, 0)

        return image, label, self.img_num

    def data_augmentation(self, image):
        # Randcom scale
        scale = tf.random_uniform([1], minval=0.5, maxval=1., dtype=tf.float32, seed=10000)
        new_h = tf.to_int32(tf.to_float(tf.shape(image)[0]) * scale)
        new_w = tf.to_int32(tf.to_float(tf.shape(image)[1]) * scale)
        image = tf.image.resize_images(image,  tf.squeeze([new_h, new_w]))

        # Random flipping
        image = tf.image.random_flip_left_right(image, seed=10000)

        return image
