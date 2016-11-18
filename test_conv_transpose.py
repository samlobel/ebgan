import tensorflow as tf
import numpy as np




# F_V = [[[[1],[1]],[[1],[1]]]]

# F_V = [[[1],[1]],[[1],[1]]]    #[[[[1],[1]],[[1],[1]]]]
# F_V = [[
#   [[1.0]],[[1.0]]],
#   [[[1.0]],[[1.0]]
# ]]
# # np_FV = np.asarray(F_V, dtype=np.float32)
# np_FV = np.ones([2,2,1,1], dtype=np.float32)
# print('FV shape: {}'.format(np_FV.shape))
# # FILTER  = tf.convert_to_tensor(F_V, dtype=tf.float32)

# # INPUT = [[
# #   [[1],[1],[1],[1]],
# #   [[1],[1],[1],[1]],
# #   [[1],[1],[1],[1]],
# #   [[1],[1],[1],[1]]]]
# np_INPUT = np.ones([1,4,4,1], dtype=np.float32)
# # np_INPUT=np.asarray(INPUT, dtype=np.float32)
# print('INPUT shape: {}'.format(np_INPUT.shape))


FILTER_SHAPE = [5,5,1,1]
INPUT_SHAPE = [1,8,8,1]


np_FILTER = np.ones(FILTER_SHAPE, dtype=np.float32)
np_INPUT = np.ones(INPUT_SHAPE, dtype=np.float32)


if __name__ == '__main__':
  with tf.Session() as sess:
    output = tf.nn.conv2d(np_INPUT, np_FILTER, strides=[1,2,2,1], padding='SAME')
    print('\n\nCONV2D')
    max_output = tf.reduce_max(output)
    norm_output = tf.div(output, max_output)
    inv_norm_output = tf.div(1.0, norm_output)
    print(inv_norm_output.eval())

  with tf.Session() as sess:
    output = tf.nn.conv2d_transpose(np_INPUT, np_FILTER, output_shape=[1,4,4,2], strides=[1,1,1,1], padding='SAME')
    print('\n\nCONV2D_TRANSPOSE')
    max_output = tf.reduce_max(output)
    norm_output = tf.div(output, max_output)   
    print(norm_output.eval())




