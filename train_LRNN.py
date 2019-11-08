import os
import time
import numpy as np
from scipy import misc
import tensorflow as tf
import data_LRNN as data
import models.LRNN as net

# GPU selection
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Important Hyper-parameters
start_lr = 1e-4
num_epoch = 100
np.random.seed(10000)
model_name = 'LRNN'
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
tr_data_reader = data.DataReader(tr_data_dir, True)
tr_input, tr_label, tr_img_num = tr_data_reader.read_data(low)

# Inference, loss, optimization
lr = tf.placeholder(tf.float32)
tr_infer = net.net(tr_input, tf.AUTO_REUSE)
loss = tf.reduce_mean(tf.abs(tr_infer - tr_label))
opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Metrics
tr_psnr =  tf.reduce_mean(tf.image.psnr(tf.to_int32(tf.clip_by_value(tr_infer, 0., 1.) * (2**high - 1)), tf.to_int32(tr_label * (2**high - 1)), max_val=(2**high - 1)))
tr_ssim =  tf.reduce_mean(tf.image.ssim(tf.to_int32(tf.clip_by_value(tr_infer, 0., 1.) * (2**high - 1)), tf.to_int32(tr_label * (2**high - 1)), max_val=(2**high - 1)))
tr_mse =  tf.reduce_mean(tf.reduce_mean(tf.square(tf.clip_by_value(tr_infer, 0., 1.) - tr_label)))

# Testing
# Data reading
te_data_reader = data.DataReader(te_data_dir, False)
te_input, te_label, te_img_num = te_data_reader.read_data(low)

# Inference, metrics
te_infer = net.net(te_input, tf.AUTO_REUSE)
te_psnr = tf.image.psnr(tf.to_int32(tf.clip_by_value(te_infer, 0., 1.) * (2**high - 1)), tf.to_int32(te_label * (2**high - 1)), max_val=(2**high - 1))
te_ssim = tf.image.ssim(tf.to_int32(tf.clip_by_value(te_infer, 0., 1.) * (2**high - 1)), tf.to_int32(te_label * (2**high - 1)), max_val=(2**high - 1))
te_mse = tf.reduce_mean(tf.square(tf.clip_by_value(te_infer, 0., 1.) - te_label))

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
    t_l, t_p, t_s, t_m, t_t = 0, 0, 0, 0, 0
    for i in range(1, t_i+1):
        start = time.time()
        _, l, p, s, m = sess.run([opt, loss, tr_psnr, tr_ssim, tr_mse], feed_dict={lr: current_lr, low: [np.random.randint(3, 7)], high: [16]})
        end = time.time()
        t_l, t_p, t_s, t_m, t_t = t_l + l, t_p + p, t_s + s, t_m + m, t_t + end - start
        if i % 100 == 0:
            print('Iter %d/%d, Loss: %f, PSNR: %f, SSIM: %f, MSE: %f, Time per Img: %f' % (i, t_i, l, p, s, m, t_t/i))
    print('Epoch %d, Loss: %f, PSNR: %f, SSIM: %f, MSE: %f, Time per Img: %f, Time per Epoch: %f' % (e, t_l/t_i, t_p/t_i, t_s/t_i, t_m/t_i, t_t/t_i, t_t))
    tr_log.write('Epoch %d, Loss: %f, PSNR: %f, SSIM: %f, MSE: %f, Time per Img: %f, Time per Epoch: %f\n' % (e, t_l/t_i, t_p/t_i, t_s/t_i, t_m/t_i,  t_t/t_i, t_t))
    tr_log.flush()

# Model Saving
save_path = saver.save(sess, check_path+'/model_100.ckpt')
print('Model saved in file: ' + save_path)

# Test
for l_b, h_b in [[3, 8], [4, 8], [3, 16], [4, 16], [5, 16], [6, 16]]:
    t_p_, t_s_, t_m_, t_t_ = 0, 0, 0, 0
    for i in range(1, te_img_num+1):
        start = time.time()
        te_input_, te_infer_, p_, s_, m_ = sess.run([te_input, te_infer, te_psnr, te_ssim, te_mse], feed_dict={low: [l_b], high: [h_b]})
        end = time.time()
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
