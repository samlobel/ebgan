import sugartensor as tf
import tf.sg_initalizer as init

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


def conv_and_scale(tensor, dim, size, stride, act, bn):
  # size is Kernal Size.
  # I wonder if opt passes down. Anyways...
  in_shape = [int(d) for d in tensor.get_shape()]
  filter_shape = [size, size, in_shape[-1], dim]
  strides = [1, stride, stride, 1]
  conv = tensor.sg_conv(size=size, dim=dim, stride=stride, act='linear', bn=False) #linear at first
  # Quick note: I'm not so sure that the biases should be multiplied by the scaler.. 
  scaler = make_scaling_matrix_for_conv(in_shape, filter_shape, strides)
  scaled_conv = tf.mul(conv, scaler)
  if bn:
    b = init.constant('b', dim)

  scaled_conv = scaled_conv + (b if bn else 0)
  out = scaled_conv.sg_identity(act=act) #That's the best way I could find to apply it.
  return out



tf.new_ops = {
  
}




