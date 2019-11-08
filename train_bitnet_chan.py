import os
import time
import numpy as np
from scipy import misc
import tensorflow as tf
import data_train as data
import models.bitnet_chan as net

# GPU selection
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Important Hyper-parameters
start_lr = 1e-4
num_epoch = 100
np.random.seed(10000)
model_name = 'bitnet_chan'
log_path = './log/' + model_name
check_path = './checkpoint/' + model_name
test_result_path = './test_result/' + model_name
os.makedirs(log_path) if not os.path.exists(log_path) else None
os.makedirs(check_path) if not os.path.exists(check_path) else None
os.makedirs(test_result_path) if not os.path.exists(test_result_path) else None

# Other Parameters
tr_log = open(log_path + '/train_mit.txt', 'a')
te_log = open(log_path + '/test_mit.txt', 'a')
tr_data_dir = './dataset/mit/train/*'
te_data_dir = './dataset/mit/test/*'

# BIt depth control
low = tf.placeholder(tf.float32, [1])
high = tf.placeholder(tf.float32, [1])

# Training
# Data reading
tr_data_reader = data.DataReader(tr_data_dir, True, with_bit_info=True)
tr_input, tr_label, tr_img_num = tr_data_reader.read_data(low)

# Inference, loss, optimization
lr = tf.placeholder(tf.float32)
tr_input_ph = tf.placeholder(tf.float32, [1, None, None, 4])
tr_label_ph = tf.placeholder(tf.float32, [1, None, None, 3])
tr_infer1 = net.net(tf.concat([tr_input_ph[:, :, :, 0:1], tr_input_ph[:, :, :, 3:4]], axis=3), tf.AUTO_REUSE)
tr_infer2 = net.net(tf.concat([tr_input_ph[:, :, :, 1:2], tr_input_ph[:, :, :, 3:4]], axis=3), tf.AUTO_REUSE)
tr_infer3 = net.net(tf.concat([tr_input_ph[:, :, :, 2:3], tr_input_ph[:, :, :, 3:4]], axis=3), tf.AUTO_REUSE)
loss1 = tf.reduce_mean(tf.abs(tr_infer1 - tr_label_ph[:, :, :, 0:1]))
loss2 = tf.reduce_mean(tf.abs(tr_infer2 - tr_label_ph[:, :, :, 1:2]))
loss3 = tf.reduce_mean(tf.abs(tr_infer3 - tr_label_ph[:, :, :, 2:3]))
opt1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss1)
opt2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss2)
opt3 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss3)

# Testing
# Data reading
te_data_reader = data.DataReader(te_data_dir, False)
te_input, te_label, te_img_num = te_data_reader.read_data(low)

# Inference
te_input_ph = tf.placeholder(tf.float32, [1, None, None, 4])
te_infer1 = net.net(tf.concat([te_input_ph[:, :, :, 0:1], te_input_ph[:, :, :, 3:4]], axis=3), tf.AUTO_REUSE)
te_infer2 = net.net(tf.concat([te_input_ph[:, :, :, 1:2], te_input_ph[:, :, :, 3:4]], axis=3), tf.AUTO_REUSE)
te_infer3 = net.net(tf.concat([te_input_ph[:, :, :, 2:3], te_input_ph[:, :, :, 3:4]], axis=3), tf.AUTO_REUSE)

# Metrics
te_infer_ph = tf.placeholder(tf.float32, [1, None, None, 3])
te_label_ph = tf.placeholder(tf.float32, [1, None, None, 3])
te_psnr = tf.image.psnr(tf.to_int32(tf.clip_by_value(te_infer_ph, 0., 1.) * (2**high - 1)), tf.to_int32(te_label_ph * (2**high - 1)), max_val=(2**high - 1))
te_ssim = tf.image.ssim(tf.to_int32(tf.clip_by_value(te_infer_ph, 0., 1.) * (2**high - 1)), tf.to_int32(te_label_ph * (2**high - 1)), max_val=(2**high - 1))
te_mse = tf.reduce_mean(tf.square(tf.clip_by_value(te_infer_ph, 0., 1.) - te_label_ph))

# Session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

# Create Saver and Restore Model
saver = tf.train.Saver(max_to_keep=1000)

# Training and Testing
t_i = tr_img_num
current_lr = start_lr
for e in range(1, num_epoch+1):
    # Learning rate decay
    if e == 76:
        current_lr /= 10

    # Training
    print('\nTraining')
    t_l, t_t = 0, 0
    for i in range(1, t_i+1):
        # Read input, label
        l_b = np.random.randint(3, 7)
        tr_input_, tr_label_ = sess.run([tr_input, tr_label], feed_dict={low: [l_b]})

        # Generate R, G, B output
        start = time.time()
        _, l1 = sess.run([opt1, loss1], feed_dict={lr: current_lr, tr_input_ph: tr_input_, tr_label_ph: tr_label_})
        _, l2 = sess.run([opt2, loss2], feed_dict={lr: current_lr, tr_input_ph: tr_input_, tr_label_ph: tr_label_})
        _, l3 = sess.run([opt3, loss3], feed_dict={lr: current_lr, tr_input_ph: tr_input_, tr_label_ph: tr_label_})
        end = time.time()

        # Logging
        t_l, t_t = t_l + l1 + l2 + l3, t_t + end - start
        if i % 100 == 0:
            print('Iter %d/%d, Loss: %f, Time per Img: %f' % (i, t_i, l1 + l2 + l3, t_t/i))
    print('Epoch %d, Loss: %f, Time per Img: %f, Time per Epoch: %f' % (e, t_l/t_i, t_t/t_i, t_t))
    tr_log.write('Epoch %d, Loss: %f, Time per Img: %f, Time per Epoch: %f\n' % (e, t_l/t_i, t_t/t_i, t_t))
    tr_log.flush()

# Model Saving
save_path = saver.save(sess, check_path+'/model_100.ckpt')
print('Model saved in file: ' + save_path)

# Test
for l_b, h_b in [[3, 8], [4, 8], [3, 16], [4, 16], [5, 16], [6, 16]]:
    t_p_, t_s_, t_m_, t_t_ = 0, 0, 0, 0
    for i in range(1, te_img_num+1):
        # Read input, label
        te_input_, te_label_ = sess.run([te_input, te_label], feed_dict={low: [l_b], high: [h_b]})

        # Generate R, G, B output
        start = time.time()
        te_infer1_ = sess.run(te_infer1, feed_dict={te_input_ph: te_input_})
        te_infer2_ = sess.run(te_infer2, feed_dict={te_input_ph: te_input_})
        te_infer3_ = sess.run(te_infer3, feed_dict={te_input_ph: te_input_})
        end = time.time()
        te_infer_ = np.concatenate((te_infer1_, te_infer2_, te_infer3_), axis=3)

        # Logging
        p_, s_, m_ = sess.run([te_psnr, te_ssim, te_mse], feed_dict={te_infer_ph: te_infer_, te_label_ph: te_label_, high: [h_b]})
        t_p_, t_s_, t_m_, t_t_ = t_p_ + p_, t_s_ + s_, t_m_ + m_, t_t_ + end - start
        input_image = np.uint8(np.squeeze(te_input_)*255.)
        infer_image = np.uint8(np.minimum(np.maximum(np.squeeze(te_infer_), 0.0), 1.)*255.)
        misc.imsave(test_result_path+'/%d_%d_%d_1_input.png' % (l_b, h_b, i), input_image)
        misc.imsave(test_result_path+'/%d_%d_%d_2_infer.png' % (l_b, h_b, i), infer_image)
    print('Low: %d, High: %d, PSNR: %f, SSIM: %f, MSE: %f, Time per Img: %f' % (l_b, h_b, t_p_/te_img_num, t_s_/te_img_num, t_m_/te_img_num, t_t_/te_img_num))
    te_log.write('Low: %d, High: %d, PSNR: %f, SSIM: %f, MSE: %f, Time per Img: %f\n' % (l_b, h_b, t_p_/te_img_num, t_s_/te_img_num, t_m_/te_img_num, t_t_/te_img_num))
    te_log.flush()

# Finish and save
coord.request_stop()
coord.join(threads)
tr_log.close()
te_log.close()
