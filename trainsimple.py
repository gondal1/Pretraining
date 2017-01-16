"""
Simple tester for the vgg16_trainable
"""

import tensorflow as tf

import vgg16_trainable as vgg16
import os
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PRETRAINED_MODEL_PATH= None
N_EPOCHS = 300
INIT_LEARNING_RATE = 0.01
WEIGHT_DECAY_RATE = 0.0005
MOMENTUM = 0.9
IMAGE_HEIGHT  = 224    #960
IMAGE_WIDTH   = 224    #720
NUM_CHANNELS  = 3
BATCH_SIZE = 50
N_CLASSES = 2
DROPOUT = 0.50
ckpt_dir = "/home/sik4hi/ckpt_dir"
LOGS_PATH = '/home/sik4hi/tensorflow_logs'
WEIGHT_PATH = '.npy'
TRAINSET_PATH = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata1.csv'
TRAINSET_PATH1 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata2.csv'
TRAINSET_PATH2 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata3.csv'
TRAINSET_PATH3 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata4.csv'
TRAINSET_PATH4 = '/mnt/data1/imagenet-data/csv-files/train/imagenetdata5.csv'
VALSET_PATH ='/mnt/data1/imagenet-data/csv-files/val/imagenetdata1.csv'

#=======================================================================================================
# Reading Training data from CSV FILE
#=======================================================================================================

csv_path = tf.train.string_input_producer([TRAINSET_PATH, TRAINSET_PATH1, TRAINSET_PATH2, TRAINSET_PATH3, TRAINSET_PATH4], shuffle=True)
textReader = tf.TextLineReader()
_, csv_content = textReader.read(csv_path)
im_name, im_label = tf.decode_csv(csv_content, record_defaults=[[""], [1]])

im_content = tf.read_file(im_name)
train_image = tf.image.decode_jpeg(im_content, channels=3)
train_image = tf.cast(train_image, tf.float32)/255. # necessary for mapping rgb channels from 0-255 to 0-1 float.
# train_image = augment(train_image)
size = tf.cast([IMAGE_HEIGHT, IMAGE_WIDTH], tf.int32)
train_image = tf.image.resize_images(train_image, size)
train_label = tf.cast(im_label, tf.int64) # unnecessary
train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label], batch_size=BATCH_SIZE,
                                                              capacity = 1000 + 3*BATCH_SIZE, min_after_dequeue = 1000)
#train_label_batch = tf.one_hot(train_label_batch, 1000)

with tf.device('/gpu:0'):
    sess = tf.Session()
    learning_rate = tf.placeholder(tf.float32, [])
    images_tf = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels_tf = tf.placeholder(tf.int64)
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16()
    vgg.build(images_tf, train_mode)
    weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
    for x in xrange(len(weights_only)):
        print (weights_only[x].name)
    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print vgg.get_var_count()

    #==============================================================================================================
    # Defining Loss, could be changed from cross entropy depending on needs. The current configuration works well on
    # multiclass (not hot-encoded vectors) prediction like ImageNET.
    #==============================================================================================================
    with tf.name_scope('Loss'):
        loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(vgg.prob, labels_tf), name='loss_tf')
        #loss_summary = tf.summary.scalar("loss", loss_tf)
        weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables())
        weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * WEIGHT_DECAY_RATE
        loss_tf += weight_decay

    # ==============================================================================================================
    # Optimizer, again it can be changed to any function provided by Tensorflow. You can simply use commented out line
    # instead of explicitly computing gradients, if you are not interested in creating summaries of gradients.
    # ==============================================================================================================
    train_op = tf.train.MomentumOptimizer(learning_rate, MOMENTUM).minimize(loss_tf)
    #optimizer = tf.train.MomentumOptimizer(0.01, MOMENTUM)
    #grads_and_vars = optimizer.compute_gradients(loss_tf)
    #grads_and_vars = map(
     #   lambda gv: (gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0] * 0.1, gv[1]),
      #  grads_and_vars)
    #grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
    #train_op = optimizer.apply_gradients(grads_and_vars)
    #===================================================================================================================
    # Summaries for the gradients
    #===================================================================================================================
    #for var in tf.trainable_variables():
     #   tf.summary.histogram(var.op.name, var)
    #summary_op = tf.summary.merge_all()


    # ===================================================================================================================
    # Accuracy for the current batch
    # ===================================================================================================================
    correct_pred = tf.equal(tf.argmax(vgg.prob, 1), labels_tf)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # ===================================================================================================================
    # Saver Operation to save and restore all variables.
    # ===================================================================================================================

with tf.device('/gpu:0'):
    sess.run(tf.initialize_all_variables())

    # For populating queues with batches, very important!
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    loss_list, valacc_list, trainacc_list, plot_loss = [], [], [], []
    #summary_writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())
    steps = 1
    count = 1

    for epoch in range(N_EPOCHS):

        train_correct = 0
        train_data = 0
        epoch_start_time = time.time()

        for i in range(13000 / BATCH_SIZE+1):
            train_imbatch, train_labatch = sess.run([train_image_batch, train_label_batch])
            _, loss_val, output_val, train_accuracy = sess.run(
                [train_op, loss_tf, vgg.prob, accuracy],
                feed_dict={learning_rate: INIT_LEARNING_RATE, images_tf: train_imbatch, labels_tf:
                    train_labatch, train_mode: True})
            loss_list.append(loss_val)
            trainacc_list.append(train_accuracy)

            train_data += len(output_val)

            if (steps) % 5 == 0:  # after 5 batches
                print "======================================"
                print "Epoch", epoch + 1, "Iteration", steps
                print "Processed", train_data, '/', 13000  # (count*BATCH_SIZE)
                print 'Accuracy: ', np.mean(trainacc_list)
                #print 'labels: ', train_labatch
                print "Training Loss:", np.mean(loss_list)
                #summary_writer.add_summary(summary_str, steps)
                loss_list = []
                trainacc_list = []
            steps += 1
            count += 1
        count = 1
        #INIT_LEARNING_RATE *= 0.99

        # for i in range(100 / BATCH_SIZE + 1):
        #     val_imbatch, val_labatch = sess.run([train_image_batch, train_label_batch])
        #     val_accuracy = sess.run(accuracy, feed_dict={images_tf: val_imbatch, labels_tf: val_labatch, train_mode: False})
        #
        # #f_log.write('epoch:' + str(epoch + 1) + '\tacc:' + str(val_accuracy) + '\n')
        # print "===========**VALIDATION ACCURACY**================"
        # print 'epoch:' + str(epoch + 1) + '\tacc:' + str(val_accuracy) + '\n'
        # print 'Time Elapsed for Epoch:' + str(epoch + 1) + ' is ' + str(
        #     (time.time() - epoch_start_time) / 60.) + ' minutes'
        # print 'Time Elapsed for Epoch:' + str(epoch + 1) + ' is ' + str(
        #     (time.time() - epoch_start_time) / 60.) + ' minutes'

    # test save
    #vgg.save_npy(sess, './test-save.npy')
