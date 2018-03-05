import helper
import os
import numpy as np
import tensorflow as tf
import time
from glob import glob
from matplotlib import pyplot as plt

# directory for datasets
DATA_DIR = './data'

def extract_data():
    """
    Extract data from MNIST & CelebA datasets
    """

    helper.download_extract('mnist', DATA_DIR)
    helper.download_extract('celeba', DATA_DIR)

def display_examples(n_images):
    """
    Display an example of MNIST & CelebA
    :param n_images: number of images to display
    """

    show_n_images = n_images

    get_ipython().magic('matplotlib inline')

    mnist_images = helper.get_batch(glob(os.path.join(DATA_DIR, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
    plt.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')
    plt.show()

    mnist_images = helper.get_batch(glob(os.path.join(DATA_DIR, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
    plt.imshow(helper.images_square_grid(mnist_images, 'RGB'))
    plt.show()

def check_tf_gpu():
    """
    Checks/prints GPU & TensorFlow version
    """
    from distutils.version import LooseVersion
    import warnings
    import tensorflow as tf

    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# checks/pre-process
extract_data()
check_tf_gpu()

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """

    # input images, z input & learning rate placehoders
    inputs_real = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    learn_rate = tf.placeholder(tf.float32)

    # return tuple of inputs real, inputs z & learning rate
    return inputs_real, inputs_z, learn_rate

def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """

    # leaky ReLU slope for x < 0
    alpha = 0.2

    with tf.variable_scope('discriminator', reuse=reuse):

        # Layer 1 (Input 28x28x3): convolution, leaky ReLU
        x1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='SAME')
        relu1 = tf.maximum(alpha * x1, x1)

        # Layer 2 (Input 14x14x64): convolution, batch normalization, leaky ReLU
        x2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='SAME')
        bn2 = tf.layers.batch_normalization(x2, training=True) # change training value for inference
        relu2 = tf.maximum(alpha * bn2, bn2)

        # Layer 3 (Input 7x7x128): convolution, batch normalization, leaky ReLU
        x3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='SAME')
        bn3 = tf.layers.batch_normalization(x3, training=True) # change training value for inference
        relu3 = tf.maximum(alpha * bn3, bn3)

        # Flatten (Input 4x4x256): flatten, logits, output
        flat = tf.reshape(relu3, (-1, 4*4*256))
        logits = tf.layers.dense(flat, 1)
        output = tf.sigmoid(logits)

        # return output & logits
        return output, logits

def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """

    # leaky ReLU slope for x < 0 & keep prob for dropout
    alpha = 0.2
    keep_prob = 0.75

    # reuse variables when it is not training
    with tf.variable_scope('generator', reuse=not is_train):

        # FC Layer
        f1 = tf.layers.dense(z, 7*7*512)
        f1 = tf.reshape(f1, (-1, 7, 7, 512))
        f1 = tf.layers.batch_normalization(f1, training=is_train)
        f1 = tf.maximum(alpha * f1, f1)

        # Layer 1 (Input 7x7x512): reshape, batch normalization, leaky ReLU, dropout
        x1 = tf.layers.conv2d_transpose(f1, 256, 5, strides=1, padding='SAME')
        x1 = tf.layers.batch_normalization(x1, training=is_train)
        x1 = tf.maximum(alpha * x1, x1)
        x1 = tf.nn.dropout(x1, keep_prob=keep_prob)

        # Layer 2 (Input 7x7x256): convolution, batch normalization, leaky ReLU, dropout
        x2 = tf.layers.conv2d_transpose(x1, 128, 5, strides=2, padding='SAME')
        x2 = tf.layers.batch_normalization(x2, training=is_train)
        x2 = tf.maximum(alpha * x2, x2)
        x2 = tf.nn.dropout(x2, keep_prob=keep_prob)

        # Layer 3 (Input 14x14x128): convolution, batch normalization, leaky ReLU, dropout
        x3 = tf.layers.conv2d_transpose(x2, 64, 5, strides=1, padding='SAME')
        x3 = tf.layers.batch_normalization(x3, training=is_train)
        x3 = tf.maximum(alpha * x3, x3)
        x3 = tf.nn.dropout(x3, keep_prob=keep_prob)

        # Layer 3 (Input 14x14x64): convolution, batch normalization, leaky ReLU, dropout
        x4 = tf.layers.conv2d_transpose(x3, 32, 5, strides=1, padding='same')
        x4 = tf.layers.batch_normalization(x4, training=is_train)
        x4 = tf.maximum(alpha * x4, x4)
        x4 = tf.nn.dropout(x4, keep_prob=keep_prob)

        # Output Layer (Input 14x14x32)
        logits = tf.layers.conv2d_transpose(x3, out_channel_dim, 5, strides=2, padding='SAME')

        # Output
        output = tf.tanh(logits)

        # return output
        return output

def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """

    # generator model
    g_model = generator(input_z, out_channel_dim)

    # discriminator models
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    # discriminator loss using sigmoid cross entropy
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    d_loss = d_loss_real + d_loss_fake

    # generator loss using sigmoid cross entropy
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    # return tuple of d_loss & g_loss
    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """

    # filter the variables with the scope names
    t_v = tf.trainable_variables()
    d_variables = [v for v in t_v if v.name.startswith('discriminator')]
    g_variables = [v for v in t_v if v.name.startswith('generator')]

    # optimize using adam & minimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_variables)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_variables)

    # return tuple of discriminator & generator training optimization
    return d_train_opt, g_train_opt

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """

    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    plt.imshow(images_grid, cmap=cmap)
    plt.show()


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """

    # create placehoders, calculate loss, optimize model
    input_real, input_z, learn_rate = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)

    steps = 0
    output_channel = 3 if data_image_mode=="RGB" else 1

    # run tf session, print step/epoch & discriminator/generator loss
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                steps += 1
                batch_images *= 2.0

                # sample random noise for generator
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                # run optimizers
                _ = sess.run(d_opt, feed_dict={input_z: batch_z,
                                               input_real: batch_images,
                                               learn_rate: learning_rate})

                _ = sess.run(g_opt, feed_dict={input_z: batch_z,
                                               input_real: batch_images,
                                               learn_rate: learning_rate})

                if steps % 10 == 0:
                    # at the end of each epoch, get the losses and print them out
                    train_loss_d = d_loss.eval({input_z: batch_z,
                                                input_real:batch_images})

                    train_loss_g = g_loss.eval({input_z: batch_z,
                                                input_real:batch_images})

                    print("Epoch {}/{}...".format(epoch_i+1, epoch_count),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

                if steps % 50 == 0:
                    show_generator_output(sess, 20, input_z, output_channel, data_image_mode)


def train_mnist(epochs, batch_size, z_dim, learning_rate, beta1):
    """
    Train the GAN using the MNIST dataset
    :param epochs: number of epochs to train
    """

    mnist_dataset = helper.Dataset('mnist', glob(os.path.join(DATA_DIR, 'mnist/*.jpg')))
    with tf.Graph().as_default():
        train(epochs, batch_size, z_dim, learning_rate, beta1,
            mnist_dataset.get_batches, mnist_dataset.shape, mnist_dataset.image_mode)


def train_celeb(epochs, batch_size, z_dim, learning_rate, beta1):
    """
    Train the GAN using the CelebA dataset
    :param epochs: number of epochs to train
    """

    celeba_dataset = helper.Dataset('celeba', glob(os.path.join(DATA_DIR, 'img_align_celeba/*.jpg')))
    with tf.Graph().as_default():
        train(epochs, batch_size, z_dim, learning_rate, beta1,
            celeba_dataset.get_batches, celeba_dataset.shape, celeba_dataset.image_mode)


def run_tests():
    """
    Runs all tests from problem_unittests
    :param b: whether to run tests
    """
    import problem_unittests as t

    t.test_model_inputs(model_inputs)
    t.test_discriminator(discriminator, tf)
    t.test_generator(generator, tf)
    t.test_model_loss(model_loss)
    t.test_model_opt(model_opt, tf)


def run_face_generation(test, train_on_mnist, train_on_faces):
    # run tests
    if test:
        run_tests()

    # hyperparameters
    batch_size = 128
    z_dim = 100
    learning_rate = 0.001
    beta1 = 0.3

    # train on datasets
    if train_m:
        train_mnist(2, batch_size, z_dim, learning_rate, beta1)

    if train_f:
        train_celeb(3, batch_size, z_dim, learning_rate, beta1)


if __name__ == '__main__':
    run_face_generation(test=False, train_on_mnist=True, train_on_faces=False)
