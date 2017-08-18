import logging as log
import os, sys 

import tensorflow as tf
import numpy as np

import reader

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_epochs', 10, 'Maximum number of epochs.')
flags.DEFINE_integer('hidden_dim', 128, 'RNN hidden state size.')
flags.DEFINE_integer('max_time_steps', 20, 'Truncated backprop length.')

flags.DEFINE_integer('vocab_size', 30000, 'Vocabulary size.')
flags.DEFINE_string('vocab_data', 'vocab.pkl', 'Vocabulary file.')
flags.DEFINE_string('train_data', 'train.txt', 'Training data.')
flags.DEFINE_string('dev_data', 'dev.txt', 'Validation data.')

flags.DEFINE_string('checkpoint_dir', '/nfs/topaz/lcheung/models/tf-test',
  'Path to checkpoints directory.')

flags.DEFINE_string('output_mode', 'debug', 'debug | info')
flags.DEFINE_string('tf_log_dir', '/nfs/topaz/lcheung/tensorboard',
  'Path to store tensorboard log files.')

FLAGS = flags.FLAGS

log.basicConfig(stream=sys.stderr, level=log.INFO)

class RNN(object):
  def __init__(self):
    '''
    init with hyperparameters here
    '''
    with tf.variable_scope('Input'):
      # inputs are a sequence of token ids, target is one time step forward 
      self.data_input = tf.placeholder(tf.int32, [ FLAGS.max_time_steps ], 'x')
      self.data_target = tf.placeholder(tf.int32, [ FLAGS.max_time_steps ], 'y')
    with tf.variable_scope('Embedding'):
      # map from one-hot encoding to hidden vector 
      self.embedding = tf.get_variable(
          'W_xm', [ FLAGS.vocab_size, FLAGS.hidden_dim ], dtype=tf.float32)
      self.embedded_input = tf.nn.embedding_lookup(
          self.embedding, self.data_input, name='x_m') 
    with tf.variable_scope('RNN'):
      self.initial_hidden_state = tf.get_variable(
          'h_init', [ 1, FLAGS.hidden_dim ], dtype=tf.float32, trainable=False)
      self.input_entry = tf.get_variable(
          'W_mh', [ FLAGS.hidden_dim, FLAGS.hidden_dim ], dtype=tf.float32)
      self.recurrence = tf.get_variable(
          'W_hh', [ FLAGS.hidden_dim, FLAGS.hidden_dim ], dtype=tf.float32)
      self.recurrence_bias = tf.get_variable(
          'b_h', [ 1, FLAGS.hidden_dim ], dtype=tf.float32)


    self.build_inference()
    self.build_loss()
    self.build_optimizer()


  def build_recurrence(self, h_prev, x_m):
    return tf.tanh(tf.matmul(tf.expand_dims(x_m, 0), self.input_entry)
                 + tf.matmul(h_prev, self.recurrence)
                 + self.recurrence_bias)

  def build_inference(self):
    with tf.variable_scope('Inference'):
      self.output = tf.scan(self.build_recurrence, self.embedded_input, 
          initializer=self.initial_hidden_state)

      self.output_exit = tf.get_variable(
          'W_hx', [ FLAGS.hidden_dim, FLAGS.vocab_size ], dtype=tf.float32)
      self.output_exit_bias = tf.get_variable(
          'b_x', [ 1, FLAGS.vocab_size ], dtype=tf.float32)
    
      self.outputs_squashed = tf.reshape(self.output, [-1, FLAGS.hidden_dim])
      self.logits = tf.matmul(self.outputs_squashed, self.output_exit) \
                  + self.output_exit_bias 
      self.token_probs = tf.nn.softmax(self.logits, name='p_ts')

      self.predicted_tokens = tf.argmax(self.token_probs, axis=1,
          name='predicteds')

  def build_loss(self):
    '''
    loss function
    '''
    with tf.variable_scope('Loss'):
      # self.predicted_tokens = tf.argmax(self.token_probs, axis=1) 
      # TODO decoding?
      # https://www.tensorflow.org/api_guides/python/nn#Classification
      # NOTE: can look here to see another functions
      self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          name='losses',
          labels=self.data_target,
          logits=self.token_probs)
      self.loss = tf.reduce_mean(self.losses, name='loss')
      tf.summary.scalar("loss_smy", self.loss)
      print('Loss shape;', self.loss.shape)

  def build_optimizer(self):
    '''
    optimizer using the loss function
    '''
    with tf.variable_scope('Optimizer'):
      #tf.train.optimizer.minimize(self.loss, name='optimizer')
      self.optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
      self.optimizer.minimize(self.loss, name='minimizer')

  def _load_or_create(self, sess):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(FLAGS.checkpoint_dir))
    self.saver = tf.train.Saver()

    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      sess.run(tf.global_variables_initializer())
      self.saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model.ckpt'))

  def train(self, data):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                    log_device_placement=True,
                    gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
      self._load_or_create(sess)

      file_writer = tf.summary.FileWriter(FLAGS.tf_log_dir, sess.graph)
      summaries = tf.summary.merge_all()

      for sample in data:
        loss, summary_output = sess.run(
            [ self.loss, summaries],
            feed_dict='')

      # TODO global_step=step
      # file_writer.add_summary(tf.summary.merge_all().eval())



def get_data():
  train_data, tok_id, id_tok  = reader.prepare_data(FLAGS.train_data,
      FLAGS.vocab_data, FLAGS.vocab_size)
  dev_data, _, _ = reader.prepare_data(FLAGS.dev_data,
      FLAGS.vocab_data, FLAGS.vocab_size)

  log.debug('Train data: %s' % train_data[:2])
  log.debug('Dev data: %s' % dev_data[:2])

  return train_data, dev_data, tok_id, id_tok

def batcher(data):
  for sample in data:

def main(_):
  train_data, dev_data, tok_id, id_tok = get_data()

  model = RNN()
  model.train(train_data)

if __name__ == '__main__':
  if FLAGS.output_mode == 'debug':
    log.getLogger().setLevel(log.DEBUG)
  elif FLAGS.output_mode == 'info':
    log.getLogger().setLevel(log.INFO)
  tf.app.run()

 
