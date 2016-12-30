from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

from inception import inception_train
from inception.imagenet_data import ImagenetData
from inception import image_processing
from vgg import vgg16_trainable as vgg16
import time
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FLAGS = tf.app.flags.FLAGS

N_EPOCHS = 500
MOMENTUM = 0.9
LOGS_PATH = '/home/sik4hi/tensorflow_logs/input'
ckpt_dir = "./ckpt_dir"
TRAINSET_PATH = '/home/sik4hi/PycharmProjects/imagenet/imagenetdata21.csv'
IMAGE_HEIGHT  = 224
IMAGE_WIDTH   = 224
BATCH_SIZE = 50

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # =======================================================================================================
    # Reading Training data from tfrecords
    # =======================================================================================================
    csv_path = tf.train.string_input_producer([TRAINSET_PATH], shuffle=True)
    textReader = tf.TextLineReader()
    _, csv_content = textReader.read(csv_path)
    im_name, im_label = tf.decode_csv(csv_content, record_defaults=[[""], [1]])

    im_content = tf.read_file(im_name)
    train_image = tf.image.decode_jpeg(im_content, channels=3)
    train_image = tf.cast(train_image,
                          tf.float32) / 255.  # necessary for mapping rgb channels from 0-255 to 0-1 float.
    # train_image = augment(train_image)
    size = tf.cast([IMAGE_HEIGHT, IMAGE_WIDTH], tf.int32)
    train_image = tf.image.resize_images(train_image, size)
    train_label = tf.cast(im_label, tf.int64)  # unnecessary
    train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label], batch_size=BATCH_SIZE,
                                                                  capacity=1000 + 3 * BATCH_SIZE,
                                                                  min_after_dequeue=1000)
    train_label_batch = tf.one_hot(train_label_batch,2)
    sess = tf.Session()
#   # =======================================================================================================
#   # Placeholders for feeding data
#   # =======================================================================================================
    with tf.name_scope('input'):
        images_tf = tf.placeholder(tf.float32, [None, 224, 224, 3])
        labels_tf = tf.placeholder(tf.float32, [None, 2])
        train_mode = tf.placeholder(tf.bool)
#
#   # ==============================================================================================================
#   # Defining object for the model class, send initialization weights and variables in here if defined in class.
#   # ==============================================================================================================
#
    vgg = vgg16.Vgg16()
    vgg.build(images_tf, train_mode)
    print (vgg.get_var_count())
    # ==============================================================================================================
    # Defining Loss, could be changed from cross entropy depending on needs. The current configuration works well on
    # multiclass (not hot-encoded vectors) prediction like ImageNET.
    # ==============================================================================================================
    with tf.device('/gpu:0'):
        with tf.name_scope('Loss') as scope:
            #labels_tf = tf.to_int64(labels_tf)
            cross_entropy = tf.reduce_sum((vgg.prob - labels_tf) ** 2)
            #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(vgg.prob, labels_tf))
            #cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(vgg.prob, labels_tf))
            #loss_summary = tf.scalar_summary("loss", cross_entropy)
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            #weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
            #weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * FLAGS.learning_rate_decay_factor
            #cross_entropy += weight_decay

        # ==============================================================================================================
        # Optimizer
        # ==============================================================================================================
        with tf.name_scope('train'):
            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0), trainable=False)
            # Calculate the learning rate schedule.
            num_batches_per_epoch = 2#(dataset_train.num_examples_per_epoch() /
                                    #FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                           global_step,
                                           decay_steps,
                                           FLAGS.learning_rate_decay_factor,
                                           staircase=True)
            # Create an optimizer that performs gradient descent.
            #train_op = tf.train.MomentumOptimizer(lr, MOMENTUM).minimize(cross_entropy)
            train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
        # ===================================================================================================================
        # Accuracy for the current batch
        # ===================================================================================================================
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_pred = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(labels_tf,1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # ===================================================================================================================
        # Saver Operation to save and restore all variables.
        # ===================================================================================================================
    with tf.device('/cpu:0'):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        saver = tf.train.Saver(max_to_keep=5)
        #f = open('out.txt', 'wb')
        #summaries.extend(input_summaries)
        #summary_op = tf.merge_summary(summaries)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(LOGS_PATH, sess.graph)
    with tf.device('/gpu:0'):
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess=sess)
        loss_list, plot_trainAccuracy, plot_loss, plot_valAccuracy = [], [], [], []
        steps = 1
        count = 1
        #im,la = sess.run([train_image_batch, train_label_batch])
        #for ax in xrange(50):
         #   mpimg.imsave("img"+str(ax)+str(la[ax]),im[ax])

        for epoch in range(N_EPOCHS):
            train_correct = 0
            train_data = 0
            epoch_start_time = time.time()
            x=int(100/FLAGS.batch_size)
            for i in range(x + 1):  # You can simply put a number here, but you definitely need to know the                                                         # size of training set.
                train_imbatch, train_labatch = sess.run([train_image_batch, train_label_batch])
                #print(train_labatch)
                _, loss_val, output_val, train_accuracy, summary_str = sess.run([train_op, cross_entropy, vgg.prob, accuracy, summary_op],
                                                                            feed_dict={images_tf: train_imbatch, labels_tf:
                                                                            train_labatch, train_mode: True})
                loss_list.append(loss_val)
                plot_trainAccuracy.append(train_accuracy)  # For visualizing training accuracy curve
                plot_loss.append(loss_val)  # For visualizing training loss curve

                train_data += len(output_val)

                if (steps) % 2 == 0:  # after 5 batches
                    print ("======================================")
                    print ("Epoch", epoch + 1, "Iteration", steps)
                    print ("Processed", train_data)  # (count*BATCH_SIZE)
                    print ('Accuracy: ', train_accuracy)
                    print ("Training Loss:", np.mean(loss_list))

                    loss_list = []
                summary_writer.add_summary(summary_str, steps)
                steps += 1
                count += 1
            if (epoch % 5 == 0):
                saver.save(sess, ckpt_dir + "/model.ckpt", global_step=epoch)
                #x1 = []
                #for i in range(len(plot_loss)):
                #    x1.append(i)
                #fig = plt.plot(x1, plot_loss)
                #fig.savefig('/home/sik4hi/loss_plot.png')  # save the figure to file
                #plt.close(fig)
                #plot_loss = []
            count = 1
            #print("Training Accuracy per Epoch:", np.mean(plot_trainAccuracy))
            #plot_trainAccuracy=[]
            # x=int(dataset_val.num_examples_per_epoch()/FLAGS.batch_size)
            # for i in range(x + 1):
            #     val_imbatch, val_labatch = sess.run([val_image_batch, val_label_batch])
            #     val_accuracy = sess.run(accuracy, feed_dict={images_tf: val_imbatch, labels_tf: val_labatch, train_mode: False})
            #
            # print ("===========**VALIDATION ACCURACY**================")
            # print ('epoch:' + str(epoch + 1) + '\tacc:' + str(val_accuracy) + '\n')
            #print ('Time Elapsed for Epoch:' + str(epoch + 1) + ' is ' + str((time.time() - epoch_start_time) / 60.) + ' minutes')
            # plot_valAccuracy.append(val_accuracy)
            # f.write('epoch:' + str(epoch + 1) + '\tacc:' + str(val_accuracy) + '\n')
            # f.write('Time Elapsed for Epoch:' + str(epoch + 1) + ' is ' + str((time.time() - epoch_start_time) / 60.) + ' minutes')  # or f.write('...\n')
            # f.write('\n')

        #f.close()
        summary_writer.close()
if __name__ == '__main__':
  tf.app.run()