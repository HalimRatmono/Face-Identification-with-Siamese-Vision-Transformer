import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops

class L1Dist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        return tf.math.abs(embedding1-embedding2)

class L1Dist_mod(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        return tf.math.reduce_sum(tf.math.abs(embedding1-embedding2), axis=1, keepdims=True)

class L2Dist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        sum_square = tf.math.reduce_sum(tf.math.square(embedding1 - embedding2), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

class L2Dist_mod(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        sum_square = tf.math.square(embedding1 - embedding2)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
    
class cosine_sim(layers.Layer):
    def _init_(self, **kwargs):
        super()._init_()
        
    def call(self, embedding1, embedding2):
        normalize_a = tf.math.l2_normalize(embedding1,axis=1)        
        normalize_b = tf.math.l2_normalize(embedding2,axis=1)
        cos_similarity=tf.math.reduce_sum(tf.multiply(normalize_a,normalize_b),axis=1, keepdims=True)
        return cos_similarity
 
class cosine_sim_mod(layers.Layer):
    def _init_(self, **kwargs):
        super()._init_()
        
    def call(self, embedding1, embedding2):
        normalize_a = tf.math.l2_normalize(embedding1,axis=1)        
        normalize_b = tf.math.l2_normalize(embedding2,axis=1)
        cos_similarity = tf.multiply(normalize_a,normalize_b)
        return cos_similarity

class TF_L2Dist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, embedding1, embedding2):
        return tf.math.reduce_euclidean_norm(embedding1,embedding2)
    
class ssim_dist(layers.Layer):
    def _init_(self, **kwargs):
        super()._init_()
        
    def call(self, embedding1, embedding2):

        temp1=embedding1
        temp2=embedding2
        temp1=tf.expand_dims(temp1, axis=1)
        temp2=tf.expand_dims(temp2, axis=1)
        temp1=tf.expand_dims(temp1, axis=-1)
        temp2=tf.expand_dims(temp2, axis=-1)
        ans = tf.image.ssim(temp1,temp2,1.0,filter_size=1,filter_sigma=1.5)
        ans=tf.expand_dims(ans, axis=1)

        return ans
    




def _verify_compatible_image_shapes(img1, img2):
  """Checks if two image tensors are compatible for applying SSIM or PSNR.
  This function checks if two sets of images have ranks at least 3, and if the
  last three dimensions match.
  Args:
    img1: Tensor containing the first image batch.
    img2: Tensor containing the second image batch.
  Returns:
    A tuple containing: the first tensor shape, the second tensor shape, and a
    list of control_flow_ops.Assert() ops implementing the checks.
  Raises:
    ValueError: When static shape check fails.
  """
  shape1 = img1.get_shape().with_rank_at_least(3)
  shape2 = img2.get_shape().with_rank_at_least(3)
  shape1[-3:].assert_is_compatible_with(shape2[-3:])

  if shape1.ndims is not None and shape2.ndims is not None:
    for dim1, dim2 in zip(
        reversed(shape1.dims[:-3]), reversed(shape2.dims[:-3])):
      if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
        raise ValueError('Two images are not compatible: %s and %s' %
                         (shape1, shape2))

  # Now assign shape tensors.
  shape1, shape2 = array_ops.shape_n([img1, img2])

  # TODO(sjhwang): Check if shape1[:-3] and shape2[:-3] are broadcastable.
  checks = []
  checks.append(
      control_flow_ops.Assert(
          math_ops.greater_equal(array_ops.size(shape1), 3), [shape1, shape2],
          summarize=10))
  checks.append(
      control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(shape1[-3:], shape2[-3:])),
          [shape1, shape2],
          summarize=10))
  return shape1, shape2, checks


def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):
  r"""Helper function for computing SSIM.
  SSIM estimates covariances with weighted sums.  The default parameters
  use a biased estimate of the covariance:
  Suppose `reducer` is a weighted sum, then the mean estimators are
    \mu_x = \sum_i w_i x_i,
    \mu_y = \sum_i w_i y_i,
  where w_i's are the weighted-sum weights, and covariance estimator is
    cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
  with assumption \sum_i w_i = 1. This covariance estimator is biased, since
    E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
  For SSIM measure with unbiased covariance estimators, pass as `compensation`
  argument (1 - \sum_i w_i ^ 2).
  Args:
    x: First set of images.
    y: Second set of images.
    reducer: Function that computes 'local' averages from the set of images. For
      non-convolutional version, this is usually tf.reduce_mean(x, [1, 2]), and
      for convolutional version, this is usually tf.nn.avg_pool2d or
      tf.nn.conv2d with weighted-sum kernel.
    max_val: The dynamic range (i.e., the difference between the maximum
      possible allowed value and the minimum allowed value).
    compensation: Compensation factor. See above.
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).
  Returns:
    A pair containing the luminance measure, and the contrast-structure measure.
  """

  c1 = (k1 * max_val)**2
  c2 = (k2 * max_val)**2

  # SSIM luminance measure is
  # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
  mean0 = reducer(x)
  mean1 = reducer(y)
  num0 = mean0 * mean1 * 2.0
  den0 = math_ops.square(mean0) + math_ops.square(mean1)
  luminance = (num0 + c1) / (den0 + c1)

  # SSIM contrast-structure measure is
  #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
  # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
  #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
  #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
  num1 = reducer(x * y) * 2.0
  den1 = reducer(math_ops.square(x) + math_ops.square(y))
  c2 *= compensation
  cs = (num1 - num0 + c2) / (den1 - den0 + c2)

  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs

def _fspecial_gauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  size = ops.convert_to_tensor(size, dtypes.int32)
  sigma = ops.convert_to_tensor(sigma)

  coords = math_ops.cast(math_ops.range(size), sigma.dtype)
  coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0

  g = math_ops.square(coords)
  g *= -0.5 / math_ops.square(sigma)

  g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
  g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
  g = nn_ops.softmax(g)
  return array_ops.reshape(g, shape=[size, size, 1, 1])

def _ssim_per_channel(img1,
                      img2,
                      max_val=1.0,
                      filter_size=11,
                      filter_sigma=1.5,
                      k1=0.01,
                      k2=0.03,
                      return_index_map=False):
  """Computes SSIM index between img1 and img2 per color channel.
  This function matches the standard SSIM implementation from:
  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
  quality assessment: from error visibility to structural similarity. IEEE
  transactions on image processing.
  Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.
  Args:
    img1: First image batch.
    img2: Second image batch.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).
    return_index_map: If True returns local SSIM map instead of the global mean.
  Returns:
    A pair of tensors containing and channel-wise SSIM and contrast-structure
    values. The shape is [..., channels].
  """
  filter_size = constant_op.constant(filter_size, dtype=dtypes.int32)
  filter_sigma = constant_op.constant(filter_sigma, dtype=img1.dtype)

  shape1, shape2 = array_ops.shape_n([img1, img2])
  checks = [
      control_flow_ops.Assert(
          math_ops.reduce_all(
              math_ops.greater_equal(shape1[-3:-1], filter_size)),
          [shape1, filter_size],
          summarize=8),
      control_flow_ops.Assert(
          math_ops.reduce_all(
              math_ops.greater_equal(shape2[-3:-1], filter_size)),
          [shape2, filter_size],
          summarize=8)
  ]

  # Enforce the check to run before computation.
  with ops.control_dependencies(checks):
    img1 = array_ops.identity(img1)

  # TODO(sjhwang): Try to cache kernels and compensation factor.
  kernel = _fspecial_gauss(filter_size, filter_sigma)
  kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

  # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
  # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
  compensation = 1.0

  # TODO(sjhwang): Try FFT.
  # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
  #   1-by-n and n-by-1 Gaussian filters instead of an n-by-n filter.
  def reducer(x):
    shape = array_ops.shape(x)
    x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
    y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    return array_ops.reshape(
        y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0))

  luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1,
                               k2)

  # Average over the second and the third from the last: height, width.
  if return_index_map:
    ssim_val = luminance * cs
  else:
    # print("***cs****")
    # print(cs)
    # tf.print(cs)
    # axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
    ssim_val = luminance * cs
    # ssim_val = math_ops.reduce_mean(luminance * cs, axes)
    # cs = math_ops.reduce_mean(cs, axes)
    ssim_val = array_ops.squeeze(ssim_val)
    print("***test****")
    print(ssim_val)
    tf.print(ssim_val)

    # print("***lu****")
    # print(luminance)
    # tf.print(luminance)
    # print("***cs****")
    # print(cs)
    # tf.print(cs)
  return ssim_val
    


class ssim_dist_mod(layers.Layer):
  def __init__(self, **kwargs):
      super().__init__()
      
  def call(self, embedding1, embedding2):
      # tf.print(embedding1)
      # tf.print(embedding1[0])
      # tf.print(embedding1[0][0])
      # tensor = tf.constant(np.zeros(16))
      # tensor=tf.expand_dims(tensor, axis=1)
      # if(embedding1.shape[0]!=None):
      # print("ssim")
      temp1=embedding1
      temp2=embedding2
      temp1=tf.expand_dims(temp1, axis=1)
      temp2=tf.expand_dims(temp2, axis=1)
      temp1=tf.expand_dims(temp1, axis=-1)
      temp2=tf.expand_dims(temp2, axis=-1)
      # print(embedding1)
      if temp1.shape[0]!=None:
          # print("s")
          ans = _ssim_per_channel(temp1,temp2,1.0,filter_size=1,filter_sigma=1.5)
      else:
          # print("l")
          ans = tf.math.abs(embedding1-embedding2)
      # ans=1-ans
      # ans=tf.expand_dims(ans, axis=1)
      # print(ans)
      # tf.print(ans)
      # else:
      #     print("l1")
      #     ans=tf.math.abs(embedding1-embedding2)
      # print(tensor)
      # tf.print(tensor)

      return ans
