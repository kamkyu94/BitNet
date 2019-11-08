import os
import glob
import time
import numpy as np
from scipy import misc
import tensorflow as tf
import data_test as data
import models.bitnet as net

# If CPU
cpu = False

# GPU selection
if cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Model name
model_name = 'bitnet'

# Session
sess = tf.InteractiveSession()

# Placeholders
low = tf.placeholder(tf.float32, [1])
high = tf.placeholder(tf.float32, [1])
full_dqimage_dirs_ph = tf.placeholder(tf.string)
image_ph = tf.placeholder(tf.float32, [1, None, None, 4])
label_ph = tf.placeholder(tf.float32, [1, None, None, 3])
infer_ph = tf.placeholder(tf.float32, [1, None, None, 3])

# Infer, Metrics
res = (2**high - 1)
infer = net.net(image_ph, tf.AUTO_REUSE)
psnr = tf.image.psnr(tf.to_int32(tf.clip_by_value(infer_ph, 0., 1.) * res), tf.to_int32(label_ph * res), max_val=res)
ssim = tf.image.ssim(tf.to_int32(tf.clip_by_value(infer_ph, 0., 1.) * res), tf.to_int32(label_ph * res), max_val=res)

# Restore Model
saver = tf.train.Saver()
saver.restore(sess, './checkpoint/' + model_name + '/model_100.ckpt')
print('Model restored')

# Run
result_dir = './test+result/' + model_name + '/'
data_dirs = glob.glob('./dataset/*')
for data_dir in data_dirs:
    # Different settings for each database
    database = data_dir.split('/')[2]
    os.makedirs(result_dir + database) if not os.path.exists(result_dir + database) else None
    if database == 'mit':
        continue
    elif database == 'espl' or database == 'kodak':
        l_b_h_b = [[3, 8], [4, 8]]
    else:
        l_b_h_b = [[3, 16], [4, 16], [5, 16], [6, 16]]

    # Call image reader
    reader = data.DataReader()
    full_dqimage_dirs, img_num = reader.read_file(data_dir+'/*')
    image, label = reader.read_data(full_dqimage_dirs_ph, database, low)

    # Logging
    if cpu:
        log = open('./log/' + model_name + '/test_'+database+'_cpu.txt', 'a')
    else:
        log = open('./log/' + model_name + '/test_' + database + '.txt', 'a')

    # Test
    for l_b, h_b in l_b_h_b:
        # Warm up
        image_, label_ = sess.run([image, label], feed_dict={full_dqimage_dirs_ph: full_dqimage_dirs[0], low: [l_b]})
        infer_ = sess.run(infer, feed_dict={image_ph: image_})

        t_p_, t_s_, t_t_ = 0, 0, 0
        for i in range(1, img_num+1):
            # Read image and label
            image_, label_ = sess.run([image, label], feed_dict={full_dqimage_dirs_ph: full_dqimage_dirs[i-1], low: [l_b]})

            # Inference, measure time
            start = time.time()
            infer_ = sess.run(infer, feed_dict={image_ph: image_})
            t_ = time.time() - start

            # Measure PSNR, SSIM
            p_, s_, = sess.run([psnr, ssim], feed_dict={high: [h_b], infer_ph: infer_, label_ph: label_})
            t_p_, t_s_, t_t_ = t_p_ + p_, t_s_ + s_, t_t_ + t_

            # Save results
            input_image = np.uint8(np.squeeze(image_) * 255.)
            infer_image = np.uint8(np.minimum(np.maximum(np.squeeze(infer_), 0.0), 1.) * 255.)
            misc.imsave(result_dir + database + '/%d_%d_%d_1_input.png' % (l_b, h_b, i), input_image)
            misc.imsave(result_dir + database + '/%d_%d_%d_2_infer.png' % (l_b, h_b, i), infer_image)

            # Logging
            print('Data:% s, Low: %d, High: %d, %d/%d, PSNR: %f, SSIM: %f, Time per Img: %f' % (database, l_b, h_b, i, img_num, p_, s_, t_))
            log.write('Data: %s, Low: %d, High: %d, %d/%d, PSNR: %f, SSIM: %f, Time per Img: %f\n' % (database, l_b, h_b, i, img_num, p_, s_, t_))
        print('Data: %s, Low: %d, High: %d, Avg PSNR: %f, SSIM: %f, Time per Img: %f' % (database, l_b, h_b, t_p_/img_num, t_s_/img_num, t_t_/img_num))
        log.write('Data: %s, Low: %d, High: %d, Avg PSNR: %f, SSIM: %f, Time per Img: %f\n\n' % (database, l_b, h_b, t_p_/img_num, t_s_/img_num, t_t_/img_num))
        log.flush()
    log.close()
