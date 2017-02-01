# Tutorial number 04 modified version, replacing PrettyTensor with TensorFlow primitives.
# GitHub link: https://github.com/miga101/tf_mnist_cnn
#
# Original author is Magnus Erik Hvass Pedersen.
# https://github.com/Hvass-Labs/TensorFlow-Tutorials

import util as util
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
from datetime import timedelta
import os


# - Constants
img_size = 28   # MNIST images are 28 by 28 pixels
# image store 1d flat array
img_size_flat = img_size * img_size
# tuple with height and width used to reshape arrays
img_shape = (img_size, img_size)
# n colors channels per image
num_channels = 1
# total number of classes
num_classes = 10

# - Define CNN: Convolutional Neural Network
# cnn layer 1
filter_size1 = 5        # Convolution filter are 5x5 pixels
num_filters1 = 16       # 16 filters at this point
# cnn layer 2
filter_size2 = 5        # Convolution filter are 5x5 pixels
num_filters2 = 36       # 36 filters at this point
# fully-connected layer
fc_size = 128           # Number of neurons in the fully-connected layer

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

# -- function to perform optimization iterations
train_batch_size = 64
# Best validation accuracy seen so far
best_validation_accuracy = 0.0
# iterations for last improvement to validation accuracy.
last_improvement = 0
# stop optimization if no improvement in this many iterations to avoid over-feeding
require_improvement = 1000
# Counter for total number of iterations performed so far.
total_iterations = 0


# # read/download the data if no present
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
print("Train set:\t{}".format(len(data.train.labels)))
print("Test set:\t{}".format(len(data.test.labels)))
print("Val set:\t{}".format(len(data.validation.labels)))

# - Class labels one-hot encoded
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)


# - Define Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# - CREATE CONVOLUTIONAL NN using prettytensor
# x_pretty = pt.wrap(x_image)
# with pt.defaults_scope(activation_fn=tf.nn.relu):
#     y_pred, loss = x_pretty.\
#         conv2d(kernel=5, depth=16, name='layer_conv1').\
#         max_pool(kernel=2, stride=2).\
#         conv2d(kernel=5, depth=36, name='layer_conv2').\
#         max_pool(kernel=2, stride=2).\
#         flatten().\
#         fully_connected(size=128, name='layer_fc1').\
#         softmax_classifier(num_classes=num_classes, labels=y_true)


# Function to define the CNN
def define_cnn(x_image=None, num_channels=None, filter_size1=None, num_filters1=None, filter_size2=None,
               num_filters2=None, fc_size=None, num_classes=None):
    # CONVOLUTIONAL LAYER 1
    layer_conv1, weights_conv1 = util.new_conv_layer(input=x_image,
                                                     num_input_channels=num_channels,
                                                     filter_size=filter_size1,
                                                     num_filter=num_filters1,
                                                     use_pooling=True)
    print(layer_conv1.shape)
    # CONVOLUTIONAL LAYER 2
    layer_conv2, weights_conv2 = util.new_conv_layer(input=layer_conv1,
                                                     num_input_channels=num_filters1,
                                                     filter_size=filter_size2,
                                                     num_filter=num_filters2,
                                                     use_pooling=True)
    print(layer_conv2.shape)
    layer_flat, num_features = util.flatten_layer(layer_conv2)
    print(layer_flat.shape)
    print(num_features)  # 1764

    # Fully-Connected Layer 1
    layer_fc1 = util.new_fc_layer(input=layer_flat, num_inputs=num_features,
                                  num_outputs=fc_size, use_relu=True)
    print(layer_fc1.shape)  # <tf.Tensor 'Relu_2:0' shape=(?, 128) dtype=float32>

    # Fully-Connected Layer 2
    layer_fc2 = util.new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes,
                                  use_relu=False)
    print(layer_fc2.shape)  # <tf.Tensor 'add_3:0' shape=(?, 10) dtype=float32>

    # Predicted Class
    y_pred = tf.nn.softmax(layer_fc2)
    # The class-number is the index of the largest element.
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    # Cost-function to be optimized
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true))

    return y_pred, y_pred_cls, loss, weights_conv1, weights_conv2


y_pred, y_pred_cls, loss, weights_conv1, weights_conv2 = define_cnn(x_image=x_image, num_channels=num_channels,
                                                                    filter_size1=filter_size1, num_filters1=num_filters1,
                                                                    filter_size2=filter_size2,
                                                                    num_filters2=num_filters2, fc_size=fc_size,
                                                                    num_classes=num_classes)
# - Optimization Method
# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
# -- Performance Measures
y_pred_cls = tf.argmax(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# - Define Saver.
saver = tf.train.Saver()
# The saved files are often called checkpoints, they may be written at regular intervals during optimization.
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')


# -- TensorFlow RUN
session = tf.Session()

# Initialize Variables
util.init_variables(session)


def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    # start time
    start_time = time.time()

    for i in range(num_iterations):

        total_iterations += 1

        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 10 iterations.
        if (i % 10 == 0) or (i == (num_iterations - 1)):
            # Calculate the accuracy on the training-batch.
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)
            # Calculate the accuracy on the training-batch.
            acc_validation, _ = validation_accuracy()
            # if improvement
            if acc_validation > best_validation_accuracy:
                best_validation_accuracy = acc_validation
                # set last improvement iteration to current
                last_improvement = total_iterations
                # Save all
                saver.save(sess=session, save_path=save_path)
                # set a mark
                improved_str = '*'
            else:
                # no improvement was found
                improved_str = ''

            # Status-message for log
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"
            # Print it.
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))
        # If no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")
            # Break out from the for-loop.
            break
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (cls_true == cls_pred)
    return correct, cls_pred


def predict_cls_test():
    return predict_cls(images = data.test.images,
                       labels = data.test.labels,
                       cls_true = data.test.cls)


def predict_cls_validation():
    return predict_cls(images = data.validation.images,
                       labels = data.validation.labels,
                       cls_true = data.validation.cls)


def validation_accuracy():
    correct, _ = predict_cls_validation()
    return cls_accuracy(correct)


def cls_accuracy(correct):
    correct_sum = correct.sum()
    acc = float(correct_sum) / len(correct)
    return acc, correct_sum


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    print('computing accuracy on test dataset...')
    correct, cls_pred = predict_cls_test()
    acc, num_correct = cls_accuracy(correct)
    num_images = len(correct)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))
    if show_example_errors:
        print("Example errors:")
        util.plot_example_errors(cls_pred=cls_pred, correct=correct, data=data, img_shape=img_shape)
    if show_confusion_matrix:
        print("Confusion Matrix:")
        util.plot_confusion_matrix(cls_pred=cls_pred, data=data, num_classes=num_classes)


# Main
print("TensorFlow version: ", tf.__version__)  # 1.0.0-rc0

# choose what to do!
train = False

if train:
    # - Train the CNN
    optimize(num_iterations=15000)
    print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)
else:
    # # - load saved CNN weights
    saver.restore(sess=session, save_path=save_path)
    print('weights loaded!')
    print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

session.close()
