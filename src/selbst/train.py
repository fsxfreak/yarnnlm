import logging as log
import os, sys, time

import tensorflow as tf
import numpy as np

import reader

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_epochs', 10, 'Maximum number of epochs.')
flags.DEFINE_integer('hidden_dim', 128, 'RNN hidden state size.')
flags.DEFINE_integer('max_time_steps', 20, 'Truncated backprop length.')

flags.DEFINE_integer('vocab_size', 30000, 'Vocabulary size.')
flags.DEFINE_string('vocab_data', 'vocab.pkl', 'Vocabulary file.')
flags.DEFINE_string('train_data', 'train.shuf.txt', 'Training data.')
flags.DEFINE_string('dev_data', 'dev.txt', 'Validation data.')

flags.DEFINE_string('checkpoint_prefix', 
  '/nfs/topaz/lcheung/models/tf-test/model',
  'Prefix of checkpoint files.')
flags.DEFINE_string('run_name', 
  'dyn_rnn',
  'Run name in tensorboard.')

flags.DEFINE_string('output_mode', 'debug', 'verbose | debug | info')
flags.DEFINE_string('tf_log_dir', '/nfs/topaz/lcheung/tensorboard',
  'Path to store tensorboard log files.')

FLAGS = flags.FLAGS

log.basicConfig(stream=sys.stderr, level=log.INFO)

def convert_id_tok(samples, id_tok):
  s_raw = [ id_tok[sample] for sample in samples ]
  return ' '.join(s_raw)

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
      # need to keep the embedded input, then feed those in as inputs
      # into the tf nn dynamic rnn cell
      self.initial_hidden_state = tf.get_variable(
          'h_init', [ 1, FLAGS.hidden_dim ], dtype=tf.float32, trainable=False,
          initializer=tf.zeros_initializer())
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
    # expand_dims used to convert 1d array to 1d vector
    return tf.tanh(tf.matmul(tf.expand_dims(x_m, 0), self.input_entry)
                 + tf.matmul(h_prev, self.recurrence)
                 + self.recurrence_bias)

  def build_inference(self):
    with tf.variable_scope('Inference'):
      self.hidden_output = tf.scan(self.build_recurrence, self.embedded_input, 
          initializer=self.initial_hidden_state)

      self.output_exit = tf.get_variable(
          'W_hx', [ FLAGS.hidden_dim, FLAGS.vocab_size ], dtype=tf.float32)
      self.output_exit_bias = tf.get_variable(
          'b_x', [ 1, FLAGS.vocab_size ], dtype=tf.float32)
    
      self.outputs_squashed = tf.reshape(self.hidden_output, [-1, FLAGS.hidden_dim])
      self.logits = tf.matmul(self.outputs_squashed, self.output_exit) \
                  + self.output_exit_bias 
      self.token_probs = tf.nn.softmax(self.logits, name='p_ts')

      self.predicted_tokens = tf.argmax(self.token_probs, axis=1,
          name='predicteds')

  def build_loss(self):
    with tf.variable_scope('Loss'):
      # self.predicted_tokens = tf.argmax(self.token_probs, axis=1) 
      # TODO decoding?
      # https://www.tensorflow.org/api_guides/python/nn#Classification
      # NOTE: can look here to see another functions
      self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          name='losses',
          labels=self.data_target,
          logits=self.logits)
      self.loss = tf.reduce_mean(self.losses, name='loss')
      tf.summary.scalar("loss_smy", self.loss)
      log.debug('Loss shape; %s' % self.loss.shape)

  def build_optimizer(self):
    '''
    optimizer using the loss function
    '''
    with tf.variable_scope('Optimizer'):
      #tf.train.optimizer.minimize(self.loss, name='optimizer')
      self.optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
      self.minimizer = self.optimizer.minimize(self.loss, name='minimizer')

      tf.summary.scalar("learning_rate", self.optimizer._learning_rate)

  def _load_or_create(self, sess):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(FLAGS.checkpoint_prefix))
    self.saver = tf.train.Saver()

    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(sess, ckpt.model_checkpoint_path)
      log.debug('Model restored from %s.' % ckpt.model_checkpoint_path)
    else:
      sess.run(tf.global_variables_initializer())
      self.saver.save(sess, FLAGS.checkpoint_prefix, global_step=0)
      log.debug('Initialized new model.')

  def train(self, data, id_tok):
    verbose = FLAGS.output_mode == 'verbose'
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=verbose,
                    log_device_placement=verbose,
                    gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
      self._load_or_create(sess)

      file_writer = tf.summary.FileWriter(
          os.path.join(FLAGS.tf_log_dir, FLAGS.run_name),
          sess.graph)
      summaries = tf.summary.merge_all()

      window = FLAGS.max_time_steps
      log.info('Starting training...')
      log.debug('Training %s' % tf.trainable_variables())

      cum_loss = 0.0
      for i in range(0, len(data) - window + 1):
        source = data[i     : i + window]
        target = data[i + 1 : i + window + 1]

        if len(source) < window:
          source = np.pad(source, (0, window - len(source)), (0, reader.PAD_ID))
        if len(target) < window:
          target = np.pad(target, (0, window - len(target)), (0, reader.PAD_ID))

        _, loss, summary_output, out = sess.run(
            [ self.minimizer, self.loss, summaries,
              self.predicted_tokens ],
            feed_dict={
              self.data_input : source,
              self.data_target: target
            })

        cum_loss = loss + cum_loss
        if i % 5000 == 0:
          log.debug('Loss %s\n\ttarget: %s\n\tpredicted: %s' 
              % (cum_loss, 
                 convert_id_tok(target, id_tok), 
                 convert_id_tok(out, id_tok)))
          cum_loss = 0
          log.debug('Saved model checkpoint to %s.' % FLAGS.checkpoint_prefix)
          self.saver.save(sess, FLAGS.checkpoint_prefix, global_step=i)

        file_writer.add_summary(summary_output, global_step=i)

def get_data():
  train_data, tok_id, id_tok  = reader.prepare_data(FLAGS.train_data,
      FLAGS.vocab_data, FLAGS.vocab_size)
  dev_data, _, _ = reader.prepare_data(FLAGS.dev_data,
      FLAGS.vocab_data, FLAGS.vocab_size)

  log.debug('Train data: %s' % train_data[:2])
  log.debug('Dev data: %s' % dev_data[:2])

  return train_data, dev_data, tok_id, id_tok

def resize_data(data):
  '''
  Squash all sentences together.
  '''
  squashed = np.concatenate(data)
  return squashed

def main(_):
  train_data, dev_data, tok_id, id_tok = get_data()

  train_data = resize_data(train_data)

  model = RNN()
  model.train(train_data, id_tok)

if __name__ == '__main__':
  if FLAGS.output_mode == 'debug' or FLAGS.output_mode == 'verbose':
    log.getLogger().setLevel(log.DEBUG)
  elif FLAGS.output_mode == 'info':
    log.getLogger().setLevel(log.INFO)
  tf.app.run()

 
