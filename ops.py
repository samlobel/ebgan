import sugartensor as tf
import numpy as np
# init = tf.sg_initializer
# sg_act = tf.sg_activation
from sugartensor import sg_initializer as init
from sugartensor import sg_activation as sg_act
import matplotlib.pyplot as plt

def get_sg_act(act_name):
  return getattr(sg_act, 'sg_' + act_name.lower())


def make_scaling_matrix_for_conv(input_shape, filter_shape, strides, padding='SAME'):
  INPUT_ONES = np.ones(input_shape, dtype=np.float32)
  FILTER_ONES = np.ones(filter_shape, dtype=np.float32)
  output = tf.nn.conv2d(INPUT_ONES, FILTER_ONES, strides=strides, padding=padding)
  max_output = tf.reduce_max(output)
  norm_output = tf.div(output, max_output)
  inv_norm_output = tf.div(1.0, norm_output)
  return inv_norm_output

def make_scaling_matrix_for_conv_transpose(input_shape, filter_shape, output_shape, strides, padding='SAME'):
  INPUT_ONES = np.ones(input_shape, dtype=np.float32)
  FILTER_ONES = np.ones(filter_shape, dtype=np.float32)
  output = tf.nn.conv2d_transpose(INPUT_ONES, FILTER_ONES, output_shape=output_shape, strides=strides, padding=padding)
  max_output = tf.reduce_max(output)
  norm_output = tf.div(output, max_output)
  inv_norm_output = tf.div(1.0, norm_output)
  return inv_norm_output


# Why don't I want to do my own conv? It's mainly for logging purposes...
# Because his variables log and initialize well. I really don't want to give that up.

def conv_and_scale(tensor, dim, size, stride, act, bn=False, bias=None):
  # size is Kernal Size.
  # I wonder if opt passes down. Anyways...
  in_shape = [int(d) for d in tensor.get_shape()]
  filter_shape = [size, size, in_shape[-1], dim]
  strides = [1, stride, stride, 1]

  conv = tensor.sg_conv(size=size, dim=dim, stride=stride, act='linear', bn=False, bias=False) #linear at first
  scaler = make_scaling_matrix_for_conv(in_shape, filter_shape, strides)
  scaled_conv = tf.mul(conv, scaler)
  scaled_with_options = scaled_conv.sg_bypass(act=act, bn=bn, bias=bias)
  return scaled_with_options

  # if bn:
  #   b = init.constant('b', dim)
  # scaled_conv = scaled_conv + (b if bn else 0)

  # act_fun = get_sg_act(act)
  # print('act_fun: {}'.format(act_fun))
  # out = act_fun(scaled_conv)
  # return out


def upconv_and_scale(tensor, dim, size, stride, act, bn=False, bias=None):
  # size is Kernal Size.
  # I wonder if opt passes down. Anyways...
  in_shape = [int(d) for d in tensor.get_shape()]
  print('in_shape for upconv: {}'.format(in_shape))
  filter_shape = [size, size, dim, in_shape[-1]]
  print('filter_shape for upconv: {}'.format(filter_shape))
  out_shape = [in_shape[0], in_shape[1]*stride, in_shape[2]*stride, dim]
  print('out_shape for upconv: {}'.format(out_shape))
  strides = [1, stride, stride, 1]

  upconv = tensor.sg_upconv(size=size, dim=dim, stride=stride, act='linear', bn=bn) #linear at first
  # Actually, batch normalization makes no difference here... Because it works on channels and not pixels
  print('upconv_shape is {}'.format(upconv.get_shape()))
  scaler = make_scaling_matrix_for_conv_transpose(in_shape, filter_shape, out_shape, strides)
  print('scaler_shape is {}'.format(scaler.get_shape()))
  scaled_upconv = tf.mul(upconv, scaler)
  scaled_with_options = scaled_upconv.sg_bypass(act=act, bn=bn, bias=bias)
  return scaled_with_options

  # if bn:
  #   b = init.constant('b', dim)
  # scaled_upconv = scaled_upconv + (b if bn else 0)

  # act_fun = get_sg_act(act)
  # print('act_fun: {}'.format(act_fun))
  # out = act_fun(scaled_upconv) #BN HAPPENS HERE.
  # return out

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


def plot_images(imgs):
  # plot result
  _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
  for i in range(10):
      for j in range(10):
          ax[i][j].imshow(imgs[i * 10 + j], 'gray')
          ax[i][j].set_axis_off()
  plt.savefig(get_next_filename(), dpi=600)
  tf.sg_info('Sample image saved to "asset/train/sample.png"')
  plt.close()




  


if __name__ == '__main__':
  print('testing')
  print('testing on matrix: ')
  with tf.Session() as sess:
    inputs_one = [8,8,5,2]
    mat = make_scaling_matrix_for_conv(*inputs_one)
    print('tensor for inputs: {}'.format(inputs_one))
    print(mat.eval())
  print('exited')


  # http://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers is soooo great.
  # Also, https://github.com/vdumoulin/conv_arithmetic is the source.



