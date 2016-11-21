# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import ops

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size
z_dim = 50        # noise dimension
margin = 1        # max-margin for hinge loss
pt_weight = 0.1  # PT regularizer's weight

#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=batch_size)

# input images
x = data.train.image

#
# create generator
#

# random uniform seed
z = tf.random_uniform((batch_size, z_dim))

size=4
stride=2
strides = [1,stride,stride,1]


with tf.sg_context(name='generator', size=4, stride=2, act='relu', bn=True, bias=False):
    g_p1 = (z.sg_dense(dim=1024)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128)))
    g_p2 = ops.upconv_and_scale(g_p1, dim=64, size=size, stride=stride,act='relu',bn=True)
    g_p3 = ops.upconv_and_scale(g_p2, dim=1, size=size, stride=stride, act='sigmoid',bn=False)
    gen = g_p3

with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  imgs = sess.run(gen)
  ops.plot_images()
print('plotted')
exit()

#
# create discriminator
#

# create real + fake image input
xx = tf.concat(0, [x, gen])

with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu', bias=False):
    d_p1 = ops.conv_and_scale(xx, dim=64, size=size, stride=stride,act='leaky_relu', bn=False)
    d_p2 = ops.conv_and_scale(d_p1, dim=128, size=size, stride=stride,act='leaky_relu', bn=False)
    d_p3 = ops.upconv_and_scale(d_p2, dim=64, size=size, stride=stride,act='leaky_relu', bn=False)
    d_p4 = ops.upconv_and_scale(d_p3, dim=1, size=size, stride=stride, act='linear', bn=False)
    disc = d_p4

#
# pull-away term ( PT ) regularizer
#

sample = gen.sg_flatten()
nom = tf.matmul(sample, tf.transpose(sample, perm=[1, 0]))
denom = tf.reduce_sum(tf.square(sample), reduction_indices=[1], keep_dims=True)
pt = tf.square(nom/denom)
pt -= tf.diag(tf.diag_part(pt))
pt = tf.reduce_sum(pt) / (batch_size * (batch_size - 1))

#
# loss & train ops
#

# mean squared errors
mse = tf.reduce_mean(tf.square(disc - xx), reduction_indices=[1, 2, 3])
mse_real, mse_fake = mse[:batch_size], mse[batch_size:]

loss_disc = mse_real + tf.maximum(margin - mse_fake, 0)   # discriminator loss
loss_gen = mse_fake + pt * pt_weight   # generator loss + PT regularizer

train_disc = tf.sg_optim(loss_disc, lr=0.001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen, lr=0.001, category='generator')  # generator train ops


#
# add summary
#

tf.sg_summary_loss(tf.identity(loss_disc, name='disc'))
tf.sg_summary_loss(tf.identity(loss_gen, name='gen'))
tf.sg_summary_image(gen)


#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_disc, train_disc])[0]  # training discriminator
    l_gen = sess.run([loss_gen, train_gen])[0]  # training generator
    return np.mean(l_disc) + np.mean(l_gen)

# do training
small_num_batches = data.train.num_batch // 10
alt_train(log_interval=10, max_ep=10, ep_size=small_num_batches, early_stop=False)

