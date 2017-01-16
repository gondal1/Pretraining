
import matplotlib.pyplot as plt

import tensorflow as tf

import vgg16_modified as vgg16
import os
import time
import numpy as np
from IPython.display import clear_output
import sys

PRETRAINED_MODEL_PATH = "/home/sik4hi/ckpt_dir"
N_EPOCHS = 300
INIT_LEARNING_RATE = 0.01
WEIGHT_DECAY_RATE = 0.0005
MOMENTUM = 0.9
IMAGE_HEIGHT = 224  # 960
IMAGE_WIDTH = 224  # 720
NUM_CHANNELS = 3
BATCH_SIZE = 80
N_CLASSES = 1000
DROPOUT = 0.50
NUM_GPUS = 1
ckpt_dir = "/home/sik4hi/ckpt_dir"
LOGS_PATH = '/home/sik4hi/tensorflow_logs'
WEIGHT_PATH = '.npy'
TRAINSET_PATH0 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata0.csv'
TRAINSET_PATH1 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata1.csv'
TRAINSET_PATH2 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata2.csv'
TRAINSET_PATH3 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata3.csv'
TRAINSET_PATH4 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata4.csv'
TRAINSET_PATH5 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata5.csv'
TRAINSET_PATH6 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata6.csv'
TRAINSET_PATH7 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata7.csv'
TRAINSET_PATH8 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata8.csv'
TRAINSET_PATH9 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata9.csv'

VALSET_PATH0 = '/mnt/data1/imagenet-data/csv-files/val/imagenetdata0.csv'
VALSET_PATH1 = '/mnt/data1/imagenet-data/csv-files/val/imagenetdata1.csv'
VALSET_PATH2 = '/mnt/data1/imagenet-data/csv-files/val/imagenetdata2.csv'
VALSET_PATH3 = '/mnt/data1/imagenet-data/csv-files/val/imagenetdata3.csv'
VALSET_PATH4 = '/mnt/data1/imagenet-data/csv-files/val/imagenetdata4.csv'


# =======================================================================================================
# Reading Training data from CSV FILE
# =======================================================================================================
def tower_loss(scope):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    Returns:
        Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images and labels for CIFAR-10.
    csv_path = tf.train.string_input_producer([TRAINSET_PATH0, TRAINSET_PATH1
                                                  , TRAINSET_PATH2, TRAINSET_PATH3
                                                  , TRAINSET_PATH4, TRAINSET_PATH5
                                                  , TRAINSET_PATH6, TRAINSET_PATH7
                                                  , TRAINSET_PATH8, TRAINSET_PATH9
                                               ], shuffle=True)
    textReader = tf.TextLineReader()
    _, csv_content = textReader.read(csv_path)
    im_name, im_label = tf.decode_csv(csv_content, record_defaults=[[""], [1]])

    im_content = tf.read_file(im_name)
    train_image = tf.image.decode_jpeg(im_content, channels=3)

    # train_image = augment(train_image)
    # size = tf.cast([IMAGE_HEIGHT, IMAGE_WIDTH], tf.int32)
    # train_image = tf.image.resize_images(train_image, size)
    train_label = tf.cast(im_label, tf.int64)  # unnecessary

    shape = tf.shape(train_image)
    height = shape[0]
    width = shape[1]
    new_shorter_edge = tf.constant(256, dtype=tf.int32)

    height_smaller_than_width = tf.less_equal(height, width)
    new_height_and_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, _compute_longer_edge(height, width, new_shorter_edge)),
        lambda: (_compute_longer_edge(width, height, new_shorter_edge), new_shorter_edge)
    )
    size = tf.cast([new_height_and_width[0], new_height_and_width[1]], tf.int32)
    train_image = tf.image.resize_images(train_image, size)
    size = tf.cast([IMAGE_HEIGHT, IMAGE_WIDTH, 3], tf.int32)
    train_image = tf.random_crop(train_image, size)
    train_image = tf.image.random_flip_left_right(train_image)
    train_image = tf.cast(train_image, tf.float32) / 255.  # necessary for mapping rgb channels from 0-255 to 0-1 float.
    # train_label_batch = tf.one_hot(train_label_batch, 1000)
    train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label]
                                                                  , batch_size=BATCH_SIZE
                                                                  , capacity=1000 + 3 * BATCH_SIZE,
                                                                  min_after_dequeue=1000)

    # Build inference Graph.
    train_mode = tf.placeholder(tf.bool)
    vgg = vgg16.Vgg16()
    vgg.build(train_image_batch, train_mode)
    weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
    for x in xrange(len(weights_only)):
        print (weights_only[x].name)
    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print vgg.get_var_count()

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(vgg.prob, train_label_batch),
                                name='loss_tf')
    # loss_summary = tf.summary.scalar("loss", loss_tf)
    weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
    weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * WEIGHT_DECAY_RATE
    total_loss += weight_decay
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def _compute_longer_edge(height, width, new_shorter_edge):
    return tf.cast(width * new_shorter_edge / height, tf.int32)


def train():
    with tf.device('/cpu:0'):
        opt = tf.train.GradientDescentOptimizer(0.01)

        tower_grads = []
        for i in xrange(NUM_GPUS):
            i = i + 1
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    loss = tower_loss(scope)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        # summaries.append(tf.contrib.deprecated.scalar_summary('learning_rate', lr))

        # Add histograms for gradients.
        # for grad, var in grads:
        #    if grad is not None:
        #        summaries.append(
        #            tf.contrib.deprecated.histogram_summary(var.op.name + '/gradients',grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads)

        # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #    summaries.append(tf.contrib.deprecated.histogram_summary(var.op.name, var))

        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #    cifar10.MOVING_AVERAGE_DECAY, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = apply_gradient_op

        # Create a saver.
        # saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        # summary_op = tf.contrib.deprecated.merge_summary(summaries)

        # Build an initialization operation to run below.
        init = tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables())

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)


        # ===================================================================================================================
        # Accuracy for the current batch
        # ===================================================================================================================
        # correct_pred = tf.equal(tf.argmax(vgg.prob, 1), labels_tf)
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        saver = tf.train.Saver(max_to_keep=10)

        loss_list, train_list, plot_loss, plot_acc, loss_list2, val_list, plot_loss2, plot_acc2 = [], [], [], [], [], [], [], []
        # summary_writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())
        steps = 1
        count = 1

        for epoch in xrange(N_EPOCHS):

            train_correct = 0
            train_data = 0
            epoch_start_time = time.time()
            print((1281144 / BATCH_SIZE) + 1)
            # print((2600 / BATCH_SIZE) + 1)
            for i in xrange((1281144 / BATCH_SIZE) + 1):
                # train_imbatch, train_labatch = sess.run([train_image_batch, train_label_batch])
                _, train_loss = sess.run([train_op, loss])

                loss_list.append(train_loss)
                # train_list.append(train_accuracy)
                sys.stdout.write('\r' + 'iteration:' + str(i))
                sys.stdout.flush()
                # train_data += len(output_val)

            # if (steps) % 5 == 0:  # after 5 batches
            clear_output()
            # t = np.mean(train_list)
            l = np.mean(loss_list)
            # print "===========**Training ACCURACY**================"
            print "Training Loss:", l
            print "Epoch", epoch + 1  # , "Iteration", steps


def main(argv=None):  # pylint: disable=unused-argument
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train()

    # test savel
    # vgg.save_npy(sess, './test-save.npy')


if __name__ == '__main__':
    tf.app.run()