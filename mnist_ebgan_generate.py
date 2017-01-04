# -*- coding: utf-8 -*-
import sugartensor as tf
import matplotlib.pyplot as plt
import ops
import os
# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32
z_dim = 50


#
# create generator
#

size=4
stride=2
strides = [1,stride,stride,1]

# random uniform seed
z = tf.random_uniform((batch_size, z_dim))

with tf.sg_context(name='generator', size=4, stride=2, act='relu', bn=True, bias=False):
    g_p1 = (z.sg_dense(dim=1024)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128)))
    g_p2 = ops.upconv_and_scale(g_p1, dim=64, size=size, stride=stride,act='relu',bn=True)
    g_p3 = ops.upconv_and_scale(g_p2, dim=1, size=size, stride=stride, act='sigmoid',bn=False)
    gen = g_p3.sg_squeeze()
           


#
# draw samples
#
asset_folder = 'asset/train/'
def get_next_filename():
  i = 0
  while True:
    num_str = str(i).zfill(4)
    filename = 'sample{}.png'.format(num_str)
    filename = os.path.join(asset_folder, filename)
    if not os.path.isfile(filename):
      print('next filename: {}'.format(filename))
      return filename
    i += 1


with tf.Session() as sess:
    tf.sg_init(sess)
    # restore parameters
    try:
      saver = tf.train.Saver()
      saver.restore(sess, tf.train.latest_checkpoint('./asset/train/ckpt'))
    except Exception as e:
      print('No saved file...: {}'.format(e))


    # run generator
    imgs = sess.run(gen)
    num_imgs = imgs.shape[0]
    num_per_side = int(num_imgs**0.5)

    # plot result
    _, ax = plt.subplots(num_per_side, num_per_side, sharex=True, sharey=True)
    for i in range(num_per_side):
        for j in range(num_per_side):
            ax[i][j].imshow(imgs[i * num_per_side + j], 'gray')
            ax[i][j].set_axis_off()
    plt.savefig(get_next_filename(), dpi=600)
    tf.sg_info('Sample image saved to "asset/train/sample.png"')
    plt.close()
