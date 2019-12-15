from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import time
import tensorflow as tf
import jiakang
#训练设备选择 -1=CPU 0=GPU0 1=GPU1
import os
os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


#配置初始化
FLAGS = tf.app.flags.FLAGS
#模型参数存储路径
tf.app.flags.DEFINE_string('train_dir', './tmp/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
#训练迭代次数
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
#是否需要存储日志文件
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#存储日志文件频率
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

#定义训练函数
def train():
  #打开训练图表（设置好的网络框架）
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()


    #提取图片和标签用于甲亢识别模型训练
    #配置CPU设备
    with tf.device('/cpu:0'):
      images, labels = jiakang.distorted_inputs()


    #建立图表并输入训练逻辑量
    logits = jiakang.inference(images)

    # 计算损失
    loss = jiakang.loss(logits, labels)


    # 建立图标并训练模型，调用jiakang.py里的train函数，并更新权重
    train_op = jiakang.train(loss, global_step)

    #记录准确率等
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    #开启会话，训练，更新权重
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
