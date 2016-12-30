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
import os

FLAGS = tf.app.flags.FLAGS

N_EPOCHS = 500
MOMENTUM = 0.9
LOGS_PATH = '/home/sik4hi/tensorflow_logs'
ckpt_dir = "/home/sik4hi/ckpt_dir"

def main(_):
    # =======================================================================================================
    # Reading Training data from tfrecords
    # =======================================================================================================
    # dataset_train = ImagenetData(subset=FLAGS.subset)
    # assert dataset_train.data_files()
    # if tf.gfile.Exists(FLAGS.train_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.train_dir)
    # tf.gfile.MakeDirs(FLAGS.train_dir)
    dataset_train = ImagenetData(subset='validation')
    #dataset_val = ImagenetData(subset='validation')
    #
    train_image_batch, train_label_batch = image_processing.distorted_inputs(dataset_train,
                                                                            num_preprocess_threads=4)
    train_label_batch = tf.one_hot(train_label_batch, depth=1000)
    #val_image_batch, val_label_batch = image_processing.distorted_inputs(
     #   dataset_val,
     #   num_preprocess_threads=4)
    #val_label_batch = tf.one_hot(val_label_batch, depth=1000)

  # =======================================================================================================
  # Placeholders for feeding data
  # =======================================================================================================
    images_tf = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels_tf = tf.placeholder(tf.float32, [None, 1000])
    train_mode = tf.placeholder(tf.bool)

  # ==============================================================================================================
  # Defining object for the model class, send initialization weights and variables in here if defined in class.
  # ==============================================================================================================

    vgg = vgg16.Vgg16()
    vgg.build(images_tf, train_mode)
    print (vgg.get_var_count())
    # ==============================================================================================================
    # Defining Loss, could be changed from cross entropy depending on needs. The current configuration works well on
    # multiclass (not hot-encoded vectors) prediction like ImageNET.
    # ==============================================================================================================
    with tf.device('/gpu:0'):
        with tf.name_scope('Loss'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(vgg.prob, labels_tf))
            #loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(vgg.prob, labels_tf))
            #loss_summary = tf.scalar_summary("loss", cross_entropy)
            #weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
            #weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * FLAGS.learning_rate_decay_factor
            #cross_entropy += weight_decay

        # ==============================================================================================================
        # Optimizer
        # ==============================================================================================================
        #global_step = tf.get_variable(
         #   'global_step', [],
        #    initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        #num_batches_per_epoch = (dataset_train.num_examples_per_epoch() /
        #                         FLAGS.batch_size)
        #decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        # Decay the learning rate exponentially based on the number of steps.
        #lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
         #                               global_step,
         #                               decay_steps,
         #                               FLAGS.learning_rate_decay_factor,
         #                               staircase=True)

        # Create an optimizer that performs gradient descent.
        train_op = tf.train.MomentumOptimizer(0.1, MOMENTUM).minimize(cross_entropy)

        # ===================================================================================================================
        # Accuracy for the current batch
        # ===================================================================================================================
        correct_pred = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(labels_tf,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # ===================================================================================================================
        # Saver Operation to save and restore all variables.
        # ===================================================================================================================
    with tf.device('/cpu:0'):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        saver = tf.train.Saver(max_to_keep=5)
        f = open('out.txt', 'wb')

    with tf.device('/gpu:0'):
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess=sess)
        loss_list, plot_trainAccuracy, plot_loss, plot_valAccuracy = [], [], [], []
        #summary_writer = tf.train.SummaryWriter(LOGS_PATH, graph=tf.get_default_graph())
        steps = 1
        count = 1

        for epoch in range(N_EPOCHS):

            train_correct = 0
            train_data = 0
            epoch_start_time = time.time()

            x=int(dataset_train.num_examples_per_epoch()/FLAGS.batch_size)
            print(x)
            for i in range(x + 1):  # You can simply put a number here, but you definitely need to know the                                                         # size of training set.
                train_imbatch, train_labatch = sess.run([train_image_batch, train_label_batch])
                _, loss_val, output_val, train_accuracy = sess.run([train_op, cross_entropy, vgg.prob, accuracy],
                                        feed_dict={images_tf: train_imbatch, labels_tf:
                                        train_labatch, train_mode: True})
                loss_list.append(loss_val)
                plot_trainAccuracy.append(train_accuracy)  # For visualizing training accuracy curve
                plot_loss.append(loss_val)  # For visualizing training loss curve

                train_data += len(output_val)

                if (steps) % 5 == 0:  # after 5 batches
                    print ("======================================")
                    print ("Epoch", epoch + 1, "Iteration", steps)
                    print ("Processed", train_data)  # (count*BATCH_SIZE)
                    print ('Accuracy: ', train_accuracy)
                    print ("Training Loss:", loss_val)
                    print ("Training Loss: mean:", np.mean(loss_list))
                    loss_list = []
                        #summary_writer.add_summary(summary, steps)
                steps += 1
                count += 1
            count = 1
            if (epoch % 5 == 0):
                saver.save(sess, ckpt_dir + "/model.ckpt", global_step=epoch)
            # x=int(dataset_val.num_examples_per_epoch()/FLAGS.batch_size)
            # for i in range(x + 1):
            #     val_imbatch, val_labatch = sess.run([val_image_batch, val_label_batch])
            #     val_accuracy = sess.run(accuracy, feed_dict={images_tf: val_imbatch, labels_tf: val_labatch, train_mode: False})
            #
            # print ("===========**VALIDATION ACCURACY**================")
            # print ('epoch:' + str(epoch + 1) + '\tacc:' + str(val_accuracy) + '\n')
            print ('Time Elapsed for Epoch:' + str(epoch + 1) + ' is ' + str((time.time() - epoch_start_time) / 60.) + ' minutes')
            # plot_valAccuracy.append(val_accuracy)
            # f.write('epoch:' + str(epoch + 1) + '\tacc:' + str(val_accuracy) + '\n')
            # f.write('Time Elapsed for Epoch:' + str(epoch + 1) + ' is ' + str((time.time() - epoch_start_time) / 60.) + ' minutes')  # or f.write('...\n')
            # f.write('\n')
        f.close()
if __name__ == '__main__':
  tf.app.run()
