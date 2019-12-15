
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import cv2
import numpy as np
import tensorflow as tf

import jiakang
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.font_manager import FontProperties

#a=matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

import os
os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './tmp/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
#模型存储路径
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/train',
                         """Directory where to read model checkpoints.""")
# tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
#                             """How often to run the eval.""")
# tf.app.flags.DEFINE_integer('num_examples', 5251,
#                             """Number of examples to run.""")
# tf.app.flags.DEFINE_boolean('run_once', False,
#                          """Whether to run eval only once.""")


#预测函数
def predict_once(saver, logits):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/dogcat_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    top_k_op = tf.nn.in_top_k(logits, [1], 1)
    sess.run([top_k_op])
    # a = tf.nn.softmax(top_k_op)
    # print(a)
    # print(logits)
    # print(top_k_op)
    result = top_k_op.eval()[0]

    if(result):
      des="1"
    else:
      des=" 0"
    print(des)
    return des
def predict(path):

  with tf.Graph().as_default() as g:
    image = jiakang.predict_input_get_resized_image(path)
    image = tf.expand_dims(image, 0)
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = jiakang.inference(image)
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        jiakang.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    return predict_once(saver, logits)



def main(argv=None):  # pylint: disable=unused-argument
  
  if len(argv)>1:
    imagePath = argv[1];
  else:
    imagePath="./data_dir/pre/ee.jpeg"
  result = predict(imagePath)
  # result = 'reslut:' + result 
  # img = mpimg.imread(imagePath)
  # #str = title.decode('utf8')
  # cv2.putText(img, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
  #             0.7, (255,0,0), 1, cv2.LINE_AA)
  # imgplot = plt.imshow(img)
  # #font = FontProperties(fname=r"/Library/Fonts/AdobeSongStd-Light.otf", size=14)
  # #plt.title(title,fontproperties=font)
  # plt.show()



if __name__ == '__main__':
  tf.app.run()
