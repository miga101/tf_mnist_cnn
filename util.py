import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import math


# Func: plotting/showing images
def plot_images(images, cls_true, cls_pred=None, img_shape=None):
    assert len(images) == len(cls_true) == 9

    # create figure with 2x2 sub-plots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # plot images
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        # true and predicted class
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # remove ticks from plot
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# Function to define new TF Variables in the given shape, initialized w random values
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


# same for biases
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# Function to define a new Convolutional layer with input of 4d tensor
def new_conv_layer(input,               # previous layer
                   num_input_channels,  # n channels in prev. layer
                   filter_size,         # width and height of each filter
                   num_filter,          # n filters
                   use_pooling=True):    # Use 2x2 max-pooling
    # Shape of the filter-weights fir the convolution according to TF API
    shape = [filter_size, filter_size, num_input_channels, num_filter]

    # new weights filter
    weights = new_weights(shape=shape)

    # new biases, one for each filter
    biases = new_biases(length=num_filter)

    # Create the TensorFlow operation fot convolution.
    # Note the strides are set to 1 in all dimensions.
    # the first and last stride must always be 1,
    # because the first is for the image number ans the
    # last is for the input-channel.
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')

    # add bias
    layer += biases

    # When using pooling to down-sample the image res.
    if use_pooling:
        # this is a 2x2 max pooling, from 2x2 moving window take the largest value,
        # then move 2 pixels to right.
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # ReLU: Rectified linear Unit
    layer = tf.nn.relu(layer)

    ''' Note that ReLU is normally executed before the pooling,
    but since relu(max_pool(x)) == max_pool(relu(x)) we can save %75
    of relus operations by max-pooling first'''

    # Return both, resulting layer and filters-weights,
    # to plot the weights later

    return layer, weights


# function for flattening a layer
# A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected layers
# after the convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can be used as input
# to the fully-connected layer.
def flatten_layer(layer):
    # get shape of the imput layer
    layer_shape = layer.get_shape()

    # assumed layer_shape == [n_images, img_height, img_width, n_channels
    # the number of features is: img_height * img_width * n_channels
    # using TF to calculate it.
    num_features = layer_shape[1:4].num_elements()

    # reshape the layer to [n_images, n_features]
    # note that we just set the size of the second dim
    # to number of features and the size in that dim is calculated
    # so the total size of the tensor is unchanged from the reshaping
    layer_flat = tf.reshape(layer, [-1, num_features])

    # shape of the flattered layer is [n_images * img_height * img_width * n_channels]

    # Return both the flattened layer and n_features
    return layer_flat, num_features


# function for creating a new Fully-Connected Layer.
# It is assumed that the input is a 2-dim tensor of shape [num_images, num_inputs].
# The output is a 2-dim tensor of shape [num_images, num_outputs].
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# Helper-function to plot example errors
def plot_example_errors(cls_pred, correct, data, img_shape):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9], img_shape=img_shape)
    plt.show()


# Helper-function to plot confusion matrix¶
def plot_confusion_matrix(cls_pred, data, num_classes):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


# Split the data-set in batches of this size to limit RAM usage.
def predict_cls(images, labels, cls_true, x, y_true, session, y_pred_cls, batch_size=256):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def init_variables(session):
    session.run(tf.global_variables_initializer())


def predict_cls_test(data):
    return predict_cls(images = data.test.images,
                       labels = data.test.labels,
                       cls_true = data.test.cls)


def predict_cls_validation(data):
    return predict_cls(images = data.validation.images,
                       labels = data.validation.labels,
                       cls_true = data.validation.cls)


def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum


def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()

    # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)


def plot_conv_weights(weights, input_channel=0, session=None):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()











