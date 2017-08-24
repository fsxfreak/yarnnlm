import logging as log
import os, sys, time

import tensorflow as tf
import numpy as np

import reader

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.2, 'Initial learning rate.')
flags.DEFINE_float('prob_dropout', 0.3, 'Dropout probability.')
flags.DEFINE_integer('max_epochs', 500, 'Maximum number of epochs.')
flags.DEFINE_integer('hidden_dim', 256, 'RNN hidden state size.')
flags.DEFINE_integer('max_time_steps', 40, 'Truncated backprop length.')
flags.DEFINE_integer('batch_size', 128, 'Num examples per minibatch.')
flags.DEFINE_integer('hidden_layers', 2, 'Num of RNN layers.')
flags.DEFINE_integer('max_grad_norm', 10, 'Clip gradients above this norm.')

flags.DEFINE_integer('vocab_size', 40000, 'Vocabulary size.')
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

def convert_id_tok(batches, id_tok):
  s_raw = [ id_tok[sample] for batch in batches for sample in batch ]
  return ' '.join(s_raw)

class RNNLM(object):
  def __init__(self, train_data, dev_data, id_tok):
    self.train_data = train_data
    self.dev_data = dev_data
    self.id_tok = id_tok

    self.build_input()
    with tf.variable_scope('Embedding'):
      # map from one-hot encoding to hidden vector 
      self.embedding = tf.get_variable(
          'W_xm', [ FLAGS.vocab_size, FLAGS.hidden_dim ], dtype=tf.float32)
      self.embedded_input = tf.nn.embedding_lookup(
          self.embedding, self.data_input, name='x_m') 
      log.debug('Embedded input shape: %s' % self.embedded_input.shape)
    with tf.variable_scope('RNN'):
      # need to keep the embedded input, then feed those in as inputs
      # into the tf nn dynamic rnn cell
      cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_dim)
      cell = tf.nn.rnn_cell.DropoutWrapper(cell,
          output_keep_prob=(1.0 - FLAGS.prob_dropout))
      rnn_layers = tf.nn.rnn_cell.MultiRNNCell([ cell ] * FLAGS.hidden_layers)

      self.initial_state = np.zeros(
          (FLAGS.hidden_layers, 2, FLAGS.batch_size, FLAGS.hidden_dim))

      self.state = tf.placeholder(tf.float32, self.initial_state.shape,
          name='rnn_state')
      l = tf.unstack(self.state, axis=0)
      rnn_state = tuple(
          [ tf.nn.rnn_cell.LSTMStateTuple(l[i][0], l[i][1])
              for i in range(FLAGS.hidden_layers) ]
          )

      self.hidden_outputs, self.next_state = tf.nn.dynamic_rnn(
          rnn_layers, self.embedded_input, 
          initial_state=rnn_state)
      log.debug('hidden output shape: %s' % self.hidden_outputs.shape)

    self.build_inference()
    self.build_loss()
    self.build_optimizer()

  def build_input(self):
    with tf.variable_scope('Input'):
      # inputs are a sequence of token ids, target is one time step forward 
      # adapted from tensorflow ptb_word_lm tutorial
      train_tensor_raw = tf.convert_to_tensor(self.train_data)

      data_len = tf.size(train_tensor_raw)
      batch_len = data_len // FLAGS.batch_size

      # strip examples which do not divide evenly, OK for an LM
      train_tensor = tf.reshape(
          train_tensor_raw[0 : FLAGS.batch_size * batch_len],
          [ FLAGS.batch_size, batch_len ])

      epoch_size = (batch_len - 1) // FLAGS.max_time_steps
      assertion = tf.assert_positive(                                 
          epoch_size,                                                 
          message="epoch_size == 0, decrease batch_size or max_time_steps")
      with tf.control_dependencies([assertion]):                      
        epoch_size = tf.identity(epoch_size, name="epoch_size")       

      i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

      self.data_input = tf.strided_slice(
          train_tensor,
          [ 0, i * FLAGS.max_time_steps ],
          [ FLAGS.batch_size, (i + 1) * FLAGS.max_time_steps ], name='x')
      self.data_target = tf.strided_slice(
          train_tensor,
          [ 0, i * FLAGS.max_time_steps + 1 ],
          [ FLAGS.batch_size, (i + 1) * FLAGS.max_time_steps + 1], name='y')

      self.data_input.set_shape([ FLAGS.batch_size, FLAGS.max_time_steps ])
      self.data_target.set_shape([ FLAGS.batch_size, FLAGS.max_time_steps ])

  def build_inference(self):
    with tf.variable_scope('Inference'):
      # combine steps from all batches into one
      self.outputs_squashed = tf.reshape(self.hidden_outputs,
          [-1, FLAGS.hidden_dim], name='squashed_h_out')
      log.debug('shape of outputs squashed: %s' % self.outputs_squashed.shape)

      self.output_exit = tf.get_variable(
          'W_hx', [ FLAGS.hidden_dim, FLAGS.vocab_size ], dtype=tf.float32)
      self.output_exit_bias = tf.get_variable(
          'b_x', [ 1, FLAGS.vocab_size ], dtype=tf.float32)
   
      self.logits = tf.matmul(self.outputs_squashed, self.output_exit) \
                  + self.output_exit_bias 
      self.logits = tf.reshape(self.logits, 
          [ FLAGS.batch_size, FLAGS.max_time_steps, FLAGS.vocab_size ])
      log.debug('shape of logits: %s' % self.logits.shape)

      self.token_probs = tf.nn.softmax(self.logits, name='p_ts')
      log.debug('shape of token probs: %s' % self.token_probs.shape)
      self.predicted_tokens = tf.argmax(self.token_probs, axis=2,
          name='predicteds')
      log.debug('shape of predicted tokens: %s' % self.predicted_tokens.shape)

  def build_loss(self):
    with tf.variable_scope('Loss'):
      log.debug('shape of targets: %s' % self.data_target.shape)
      '''
      self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          name='losses',
          labels=self.data_target,
          logits=self.logits)
      self.loss = tf.reduce_mean(self.losses, name='loss')
      '''
      self.loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
          self.logits,
          self.data_target,
          tf.ones([ FLAGS.batch_size, FLAGS.max_time_steps], dtype=tf.float32),
          average_across_timesteps=False,
          average_across_batch=True,
          name='loss'))

      tf.summary.scalar("loss_smy", self.loss)
      log.debug('Loss shape; %s' % self.loss.shape)

  def build_optimizer(self):
    '''
    optimizer using the loss function
    '''
    with tf.variable_scope('Optimizer'):
      self.optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

      self.gradients, _ = tf.clip_by_global_norm(
          tf.gradients(self.loss, tf.trainable_variables()), 
          FLAGS.max_grad_norm)
      self.minimizer = self.optimizer.apply_gradients(
          zip(self.gradients, tf.trainable_variables()),
          global_step=tf.contrib.framework.get_or_create_global_step())
      #self.minimizer = self.optimizer.minimize(self.loss, name='minimizer')

      tf.summary.scalar("learning_rate", self.optimizer._learning_rate)

  def _load_or_create(self, sess):
    ckpt = tf.train.get_checkpoint_state(
      os.path.dirname(FLAGS.checkpoint_prefix))
    self.saver = tf.train.Saver()

    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(sess, ckpt.model_checkpoint_path)
      log.debug('Model restored from %s.' % ckpt.model_checkpoint_path)
    else:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      self.saver.save(sess, FLAGS.checkpoint_prefix, global_step=0)
      log.debug('Initialized new model.')

    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

  def validate(self, session):
    pass 

  def train(self):
    verbose = FLAGS.output_mode == 'verbose'
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=verbose,
                    log_device_placement=verbose,
                    gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
      self._load_or_create(sess)

      file_writer = tf.summary.FileWriter(
          os.path.join(FLAGS.tf_log_dir, FLAGS.run_name),
          sess.graph)
      summaries = tf.summary.merge_all()

      log.info('Starting training...')
      log.debug('Training %s' % tf.trainable_variables())

      data_len = len(self.train_data)
      num_steps = ((data_len // FLAGS.batch_size) - 1) // FLAGS.max_time_steps

      global_step = 0
      cum_loss = 0.0
      for epoch in range(FLAGS.max_epochs):
        state = self.initial_state
        for step in range(num_steps):
          _, loss, summary_output, state, target, out = sess.run(
              [ self.minimizer, self.loss, summaries,
                self.next_state,
                self.data_target,
                self.predicted_tokens ],
              feed_dict={
                self.state : state
              })

          cum_loss = loss + cum_loss
          if global_step % 1000 == 0:
            log.info('Epoch: %d, step: %d, loss %s' 
                % (epoch, global_step, cum_loss))
            log.debug('\ttarg: %s\n\tpred: %s' 
                % (convert_id_tok(target, self.id_tok)[:50],
                   convert_id_tok(out, self.id_tok)[:50]))
            cum_loss = 0
            log.debug('Saved model checkpoint to %s.' % FLAGS.checkpoint_prefix)
            self.saver.save(sess, FLAGS.checkpoint_prefix, 
                global_step=global_step)

          file_writer.add_summary(summary_output, global_step=global_step)
          global_step = global_step + 1

      self.coord.request_stop()
      self.coord.join(self.threads)

def get_data():
  train_data, tok_id, id_tok  = reader.prepare_data(FLAGS.train_data,
      FLAGS.vocab_data, FLAGS.vocab_size)
  dev_data, _, _ = reader.prepare_data(FLAGS.dev_data,
      FLAGS.vocab_data, FLAGS.vocab_size)

  train_data = reader.squash_data(train_data)
  dev_data = reader.squash_data(dev_data)

  log.debug('Train data: %s' % train_data[:10])
  log.info('Length of training data: %d' % len(train_data))
  log.debug('Dev data: %s' % dev_data[:10])
  log.info('Length of dev data: %d' % len(dev_data))

  return train_data, dev_data, tok_id, id_tok

def main(_):
  train_data, dev_data, tok_id, id_tok = get_data()

  model = RNNLM(train_data, dev_data, id_tok)
  model.train()

if __name__ == '__main__':
  if FLAGS.output_mode == 'debug' or FLAGS.output_mode == 'verbose':
    log.getLogger().setLevel(log.DEBUG)
  elif FLAGS.output_mode == 'info':
    log.getLogger().setLevel(log.INFO)
  tf.app.run()

 
