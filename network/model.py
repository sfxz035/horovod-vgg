from network.ops import *
import tensorflow as tf
mean_x = 127  # tf.reduce_mean(self.input)
mean_y = 127  # tf.reduce_mean(self.input)


#### generator
def net(input,reuse=False,is_training=True,args=None,name='SRResnet'):
    with tf.variable_scope(name,reuse=reuse):
        with tf.variable_scope('input_stage'):
            L1 = PReLU(conv_b(input,args.SPFILTER_DIM,k_h=9,k_w=9,name='conv2d_1'),name='PReLU_1')
            x = L1
            for i in range(args.nubBlocks_SR):
                x = resBlock_SR(x,args.SPFILTER_DIM,is_training=is_training,name='Block_'+str(i))
            L2 = conv_bn(x,args.SPFILTER_DIM,is_train=is_training,name='con2d_2')
            L2 = L2 +L1
        with tf.variable_scope('subpixelconv_stage1'):
            assert args.scale == 4
            L3_U1 = conv_b(L2,args.SPFILTER_DIM*4,name='connv2d_U1')
            L3_U1 = pixelShuffler(L3_U1,2)
            L3_U1 = PReLU(L3_U1,name='PReLU_U1')
            L4_U2 = conv_b(L3_U1,args.SPFILTER_DIM*4,name='conv2d_U2')
            L4_U2 = pixelShuffler(L4_U2,2)
            L4_U2 = PReLU(L4_U2,name='PReLU_U2')
        with tf.variable_scope('ouput_stage'):
            output = conv_b(L4_U2,3,k_h=9,k_w=9,name='conv2d_out')
        return output

### vgg
def vgg16(inputs,is_training=False,name='vgg_19',reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        ### layer1
        L1_1 = conv_relu(inputs,64,name='L1_1')
        L1_2 = conv_relu(L1_1,64,name='L1_2')
        pool1 = tf.nn.max_pool(L1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling1')  ##

        ### layer2
        L2_1 = conv_relu(pool1,128,name='L2_1')
        L2_2 = conv_relu(L2_1,128,name='L2_2')
        pool2 = tf.nn.max_pool(L2_2, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling1')

        ### layer3
        L3_1 = conv_relu(pool2,256,name='L3_1')
        L3_2 = conv_relu(L3_1,256,name='L3_2')
        L3_3 = conv_relu(L3_2,256,name='L3_3')
        pool3 = tf.nn.max_pool(L3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling1')

        ### layer4
        L4_1 = conv_relu(pool3,512,name='L4_1')
        L4_2 = conv_relu(L4_1,512,name='L4_2')
        L4_3 = conv_relu(L4_2,512,name='L4_3')
        pool4 = tf.nn.max_pool(L4_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling1')

        ### layer5
        L5_1 = conv_relu(pool4,512,name='L5_1')
        L5_2 = conv_relu(L5_1,512,name='L5_2')
        L5_3 = conv_relu(L5_2,512,name='L5_3')
        pool5 = tf.nn.max_pool(L5_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling1')

        flatten = tf.layers.flatten(pool5)
        FC_1 = ReLU(tf.layers.dense(flatten,4096),name='ReLU_FC1')
        FC_2 = ReLU(tf.layers.dense(FC_1,4096),name='ReLU_FC2')
        FC_3 = ReLU(tf.layers.dense(FC_2,1),name='ReLU_FC2')
        out = tf.nn.sigmoid(FC_3,name='sigmoid')
        return out

def loss(output,label,EPS=1e-12):
    log_loss = (1-label)*tf.log(1 - output + EPS)+label*tf.log(output + EPS)
    log_loss = tf.reduce_mean(-log_loss)
    # MSE_loss = tf.reduce_mean(tf.square(label - output))
    return log_loss

