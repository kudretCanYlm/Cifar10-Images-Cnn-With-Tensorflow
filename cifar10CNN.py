from asyncio.windows_events import NULL
from pandas import array
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import time as time
from datetime import timedelta
import cifar10
import os
# cifar10.download()


# tf.placeholder(tf.float32, [1, 227, 227, 3])

# you have to change it to this:

# x = tf.Variable(tf.zeros(shape=(1, 227, 227, 3)), name='x', dtype=tf.float32)
tf.compat.v1.disable_v2_behavior()

print(cifar10.load_class_names())

train_img, train_cls, train_lables = cifar10.load_training_data()
test_img, test_cls, test_lables = cifar10.load_test_data()

print("{} , {}", len(train_img), len(test_img))

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
# x=tf.Variable(tf.zeros(shape=(5000,32,32,3)),name="x",dtype=tf.float32)
y_true = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 10])
# y_true=tf.Variable(tf.zeros(shape=(5000,10)),dtype=tf.float32)
pkeep = tf.compat.v1.placeholder(dtype=tf.float32)
# pkeep=tf.Variable(0.2,dtype=tf.float32)
# batch normalizasyon (oranı arttırır)
# burada hata veriyor main 2ye bak


# data agmantation


def pre_process_image(image):
    # resmi ters çevir
    image = tf.image.random_flip_left_right(image)
    # renkleriyle oyna
    image = tf.image.random_hue(image, max_delta=0.05)
    # resimin konstrası ile oyna
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    # resimin parlaklığı ile oyna
    image = tf.image.random_brightness(image, max_delta=0.2)
    # resimin doygunluğu ile oyna
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
    image = tf.image.random_flip_left_right(image)
    # güvelik önlemleri

    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)
    print(image)
    # init = tf.compat.v1.initialize_all_variables()
    # sess = tf.compat.v1.Session()
    # sess.run(init)
    # im = sess.run(image)
    # plt.imshow(im)
    # plt.show()

    return image


def pre_process(images):
    # tüm resimlere uygulama
    images = tf.map_fn(lambda image: pre_process_image(image), images)
    return images


# aşağıdaki kısım cpu da daha hızlı çalışır
with tf.device("/cpu:0"):
    distorted_images = pre_process(images=x)


def conv_layer(input, size_in, size_out, use_pooling=True):
    w = tf.Variable(tf.compat.v1.truncated_normal(
        [3, 3, size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))

    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")+b
    y = tf.nn.relu(conv)

    if(use_pooling):
        y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding="SAME")
    return y

# fuly connected layer -Dense layer


def fc_layer(input, size_in, size_out, relu=True, dropout=True):
    w = tf.Variable(tf.compat.v1.truncated_normal(
        [size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))

    logits = tf.matmul(input, w)+b

    if(relu):
        y = tf.nn.relu(logits)
        if(dropout):
            y = tf.nn.dropout(y, pkeep)
        return y
    else:
        return logits


#          giriş değeri,kaç katman olduğu,yei verilen katman
conv1 = conv_layer(x, 3, 32, use_pooling=True)
conv2 = conv_layer(conv1, 32, 64, use_pooling=True)
conv3=conv_layer(conv2, 32, 48, use_pooling=True)
conv4 = conv_layer(conv3, 48, 64, use_pooling=True)
flattened = tf.reshape(conv4, [-1, 2*2*64])

fc1 = fc_layer(flattened, 2*2*64, 512, relu=True, dropout=True)
fc2 = fc_layer(fc1, 512, 256, relu=True, dropout=True)
logits = fc_layer(fc2, 256, 10, relu=False, dropout=False)
y = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(y, 1)
xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)

loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y_true, 1))
accuary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

global_step = tf.Variable(0, trainable=False)
optimizer = tf.compat.v1.train.AdamOptimizer(
    learning_rate=0.001).minimize(loss, global_step=global_step)

sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# kaydetme
saver = tf.compat.v1.train.Saver()
save_dir = "checkpoints/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "cifar10CNN")

try:
    print("Checkpoint Yükleniyor...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(sess, save_path=last_chk_path)
    print("Checkpoint yüklendi: ", last_chk_path)
except:
    print("checkpoint bulunamadı:")
    sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 128


def random_batch():
    index = np.random.choice(len(train_img), size=batch_size, replace=False)
    x_batch = train_img[index, :, :, :]
    y_batch = train_lables[index, :]

    return x_batch, y_batch


loss_graph = []


def training_step(iterations):
    start_time = time.time()
    acc_total = []
    for i in range(iterations):
        x_batch, y_batch = random_batch()
        feed_dict_train = {x: x_batch, y_true: y_batch, pkeep: 0.5}
        [_, train_loss, g_step] = sess.run(
            [optimizer, loss, global_step], feed_dict=feed_dict_train)
        loss_graph.append(train_loss)

        if(i % 100 == 0):
            acc = sess.run(accuary, feed_dict=feed_dict_train)
            print("Iteration: ", i, " Training accuary: ",
                  acc, " Trainning loss: ", train_loss)
            acc_total.append(acc)
        if(g_step % 1000 == 0):
            saver.save(sess, save_path, global_step=global_step)
            print("Checkpoint Kaydedildi")

    end_time = time.time()
    time_dif = end_time-start_time
    print("Time usage: ", timedelta(seconds=int(round(time_dif))))
    plt.plot(acc_total)
    plt.show()


batch_size_test = 256


def test_accuary():
    num_images = len(test_img)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0

    while (i < num_images):
        j = min(i+batch_size_test, num_images)
        feed_dict_test = {x: test_img[i:j, :],
                          y_true: test_lables[i:j, :], pkeep: 1}
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict_test)
        i = j
    correct = (test_cls == cls_pred)
    print(correct)
    print("Testing accuary: ", np.mean(correct))


training_step(5000)
test_accuary()

plt.plot(loss_graph, "k-")
plt.title("Loss Grafiği")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
