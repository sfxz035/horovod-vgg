import horovod.tensorflow as hvd
import tensorflow as tf
import os
import network.model as model
import argparse
import dataset


parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate",default=0.0001,type=float)
parser.add_argument("--batch_size",default=10,type=int)
parser.add_argument("--savenet_path",default='./libSaveNet/savenet/')
parser.add_argument("--train_file",default='./data/train')
parser.add_argument("--test_file",default='./data/test')

args = parser.parse_args()

def train():
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
    # config.gpu_options.allow_growth = True      #程序按需申请内存
    x_train, y_train = dataset.get_data(args.train_file, min=20)
    x_test, y_test = dataset.get_data(args.test_file,min=20)
    x = tf.placeholder(tf.float32, shape=[args.batch_size, 300, 300, 3])
    y_ = tf.placeholder(tf.float32, shape=[args.batch_size, 1])
    y = model.vgg16(x,name='SRResnet')
    loss = model.loss(y,y_)
    opt = tf.train.AdamOptimizer(args.learning_rate* hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    train_op = opt.minimize(loss)
    checkpoint_dir = './libSaveNet/savenet/' if hvd.rank() == 0 else None
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                          config=config,
                                          hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
            count, m = 0, 0
            for ep in range(args.epoch):
                batch_idxs = len(x_train) // args.batch_size* hvd.size()
                for idx in range(batch_idxs):
                    # batch_input = x_train[idx * args.batch_size: (idx + 1) * args.batch_size]
                    # batch_labels = y_train[idx * args.batch_size: (idx + 1) * args.batch_size]
                    batch_input, batch_labels = dataset.random_batch(x_train,y_train,args.batch_size)

                    mon_sess.run(train_op, feed_dict={x: batch_input, y_: batch_labels})
                    count += 1
                    # print(count)
                    if count % 100 == 0 and hvd.rank() == 0:
                        m += 1
                        batch_input_test, batch_labels_test = dataset.random_batch(x_test, y_test, args.batch_size)
                        # batch_input_test = x_test[0: args.batch_size]
                        # batch_labels_test = y_test[0: args.batch_size]
                        loss1 = mon_sess.run(loss, feed_dict={x: batch_input, y_: batch_labels})
                        loss2 = mon_sess.run(loss, feed_dict={x: batch_input_test, y_: batch_labels_test})
                        print("Epoch: [%2d], step: [%2d], train_loss: [%.8f]" \
                              % ((ep + 1), count, loss1), "\t",
                              'test_loss:[%.8f]' % (loss2))



