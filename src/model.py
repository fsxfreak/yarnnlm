import logging as log
import os, sys, timeit

import tensorflow as tf
import numpy as np

import reader

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.8, 'Initial momentum.')
flags.DEFINE_float('prob_dropout', 0.5, 'Dropout probability.')
flags.DEFINE_integer('max_epochs', 50, 'Maximum number of epochs.')
flags.DEFINE_integer('hidden_dim', 256, 'RNN hidden state size.')
flags.DEFINE_integer('max_time_steps', 40, 'Truncated backprop length.')
flags.DEFINE_integer('batch_size', 64, 'Num examples per minibatch.')
flags.DEFINE_integer('hidden_layers', 2, 'Num of RNN layers.')
flags.DEFINE_integer('max_grad_norm', 10, 'Clip gradients above this norm.')

flags.DEFINE_integer('vocab_size', 40000, 'Vocabulary size.')
flags.DEFINE_string('vocab_data', 'data/vocab.pkl', 'Vocabulary file.')
flags.DEFINE_string('train_data', 'data/train.shuf.txt', 'Training data.')
flags.DEFINE_string('dev_data', 'data/dev.txt', 'Validation data.')
flags.DEFINE_string('predict_data', 'data/test.txt',
    'Generate sentences for each seed line of this file.')
flags.DEFINE_string('score_data', 'data/play.txt',
    'Generate sentences for each seed line of this file.')

flags.DEFINE_string('checkpoint_prefix', 
  '/nfs/topaz/lcheung/models/tf-test/model',
  'Prefix of checkpoint files.')
flags.DEFINE_string('run_name', 
  'testrun',
  'Run name in tensorboard.')

flags.DEFINE_string('output_mode', 'debug', 'verbose|debug|info')
flags.DEFINE_string('tf_log_dir', '/nfs/topaz/lcheung/tensorboard',
  'Path to store tensorboard log files.')

FLAGS = flags.FLAGS

log.basicConfig(stream=sys.stderr, level=log.INFO,
    format='%(asctime)s [%(levelname)s]:%(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S')                        

def convert_id_tok(batches, id_tok):
  s_raw = [ id_tok[sample] for batch in batches for sample in batch 
              if sample < len(id_tok)]
  return ' '.join(s_raw)

class RNNLM(object):
  def __init__(self, train_data, dev_data, id_tok):
    self.train_data = train_data
    self.dev_data = dev_data
    self.id_tok = id_tok

    self.graph = tf.Graph()

    with self.graph.as_default():
      self.global_step = tf.Variable(0, trainable=False, name='global_step')

      self.build_input()
      self.build_embedding()
      self.build_rnn()
      self.build_inference()
      self.build_loss()
      self.build_optimizer()

  def build_input(self):
    with tf.variable_scope('Input', reuse=True):
      # one batch is spans batch_size + max_time_steps + 1, because
      # there are batch_size source/target pairs, with target shifted to
      # the right of source by 1
      self.batch_len = FLAGS.batch_size + FLAGS.max_time_steps + 1
      self.data_raw = tf.placeholder(tf.int32, shape=[ self.batch_len ], 
          name='raw_xy')

      # NOTE: samples which do not divide evenly at the end of the data will 
      # not be used
      self.batches_per_epoch = len(self.train_data) // self.batch_len
      self.batch_select = tf.train.range_input_producer(
          self.batches_per_epoch, shuffle=True, name='batch_index').dequeue()

      sliced = []
      for i in range(FLAGS.batch_size):
        sliced.append(tf.slice(self.data_raw, [i], [ FLAGS.max_time_steps ]))
      self.data_input = tf.stack(sliced, name='x')
      log.debug('Data input shape: %s' % self.data_input.shape)
      sliced = []
      for i in range(FLAGS.batch_size):
        sliced.append(tf.slice(self.data_raw, [i+1], [ FLAGS.max_time_steps ]))
      self.data_target = tf.stack(sliced, name='y')
      log.debug('Data target shape: %s' % self.data_input.shape)

  def build_embedding(self):
    with tf.variable_scope('Embedding'):
      # map from one-hot encoding to hidden vector 
      self.embedding = tf.get_variable(
          'W_xm', [ FLAGS.vocab_size, FLAGS.hidden_dim ], dtype=tf.float32)
      self.embedded_input = tf.nn.embedding_lookup(
          self.embedding, self.data_input, name='x_m') 
      log.debug('Embedded input shape: %s' % self.embedded_input.shape)

  def build_rnn(self):
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

      # self.data_target : [ batch_size, max_time_steps ]
      # self.token_probs : [ batch_size, max_time_steps, vocab_size ]
      data_target_indices = tf.stack(
        [
          tf.range(FLAGS.batch_size * FLAGS.max_time_steps),
          tf.reshape(self.data_target,
                    [ FLAGS.batch_size * FLAGS.max_time_steps ])
        ], axis=1)
      log.debug('data_target_indices: %s' % data_target_indices.shape)
      token_probs_flattened = tf.reshape(self.token_probs,
          [ FLAGS.batch_size * FLAGS.max_time_steps, FLAGS.vocab_size])
      log.debug('token_probs_flattened: %s' % token_probs_flattened.shape)
      self.score = tf.gather_nd(token_probs_flattened, data_target_indices,
          name='score')
      log.debug('shape of score probs: %s' % self.score.shape)

  def build_loss(self):
    with tf.variable_scope('Loss'):
      log.debug('shape of targets: %s' % self.data_target.shape)

      self.loss = tf.contrib.seq2seq.sequence_loss(
          self.logits,
          self.data_target,
          tf.ones([ FLAGS.batch_size, FLAGS.max_time_steps], dtype=tf.float32),
          average_across_timesteps=True,
          average_across_batch=True,
          name='loss')
      log.debug('Loss shape; %s' % self.loss.shape)

      self.perplexity = tf.pow(2, self.loss, name='perplexity')

      tf.summary.scalar("loss_smy", self.loss)
      tf.summary.scalar("perplexity", self.perplexity)

  def build_optimizer(self):
    '''
    optimizer using the loss function
    '''
    with tf.variable_scope('Optimizer'):
      self.optimizer = tf.train.MomentumOptimizer(
          FLAGS.learning_rate, FLAGS.momentum, use_nesterov=True)

      self.gradients, _ = tf.clip_by_global_norm(
          tf.gradients(self.loss, tf.trainable_variables()), 
          FLAGS.max_grad_norm)
      self.minimizer = self.optimizer.apply_gradients(
          zip(self.gradients, tf.trainable_variables()),
          global_step=self.global_step)

  def _load_or_create(self, sess):
    with self.graph.as_default():
      ckpt = tf.train.get_checkpoint_state(
        os.path.dirname(FLAGS.checkpoint_prefix))
      self.saver = tf.train.Saver()

      if ckpt and ckpt.model_checkpoint_path:
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        log.debug('Model restored from %s.' % ckpt.model_checkpoint_path)
      else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        self.saver.save(sess, FLAGS.checkpoint_prefix,
            global_step=self.global_step)
        log.debug('Initialized new model.')

      self.coord = tf.train.Coordinator()
      self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

  def _load(self, sess):
    with self.graph.as_default():
      ckpt = tf.train.get_checkpoint_state(
        os.path.dirname(FLAGS.checkpoint_prefix))
      self.saver = tf.train.Saver()

      if ckpt and ckpt.model_checkpoint_path:
        self.saver.restore(sess, ckpt.model_checkpoint_path)
        log.debug('Model restored from %s.' % ckpt.model_checkpoint_path)
      else:
        raise ValueError('Unable to find existing trained model.')

      self.coord = tf.train.Coordinator()
      self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

  def validate(self, sess):
    log.info('Running validation pass...')

    state = self.initial_state
    cum_loss = 0.0
    for i in range(len(self.dev_data) - self.batch_len):
      data_begin = i * self.batch_len
      data_end = data_begin + self.batch_len

      data = self.dev_data[data_begin : data_end]
      if len(data) != self.batch_len:
        continue

      loss, state, target, out = sess.run(
          [ self.loss, 
            self.next_state,
            self.data_target,
            self.predicted_tokens ],
          feed_dict={
            self.state : state,
            self.data_raw : self.dev_data[data_begin : data_end]
          })

      if i % 10000 == 0:
        log.debug('\n\ttarg: %s\n\tpred: %s' 
            % (convert_id_tok(target, self.id_tok)[:FLAGS.max_time_steps],
               convert_id_tok(out, self.id_tok)[:FLAGS.max_time_steps]))
      cum_loss += loss

    log.info('Validation loss: %.3f, num words: %d' 
        % (cum_loss, len(self.dev_data)))

  def predict(self, data):
    log.info('Predicting test data...')
    verbose = FLAGS.output_mode == 'verbose'
    with tf.Session(graph=self.graph,
                    config=tf.ConfigProto(allow_soft_placement=verbose,
                    log_device_placement=verbose,
                    gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
      self._load(sess)

      for line in data:
        state = self.initial_state

        source = np.pad(np.array(line), 
            (0, self.batch_len - len(line) % self.batch_len), 'constant')
        predict_index = len(line) - 2

        log.debug('Seeding with: %s' % 
            convert_id_tok([ source[:predict_index + 1] ], self.id_tok))
        probs, preds, state = sess.run(
            [ self.token_probs, self.predicted_tokens, self.next_state ],
            feed_dict={
              self.state : state,
              self.data_raw : source
              })
      
        pred_word_id = preds[0][predict_index]
        pred_word = self.id_tok[pred_word_id]
        prob_pred = probs[0][predict_index][pred_word_id] 
        log.debug('pred: %s' % pred_word)
        log.debug('prob: %f' % prob_pred)

        while pred_word != '<EOS>' and predict_index < FLAGS.max_time_steps - 1:
          source[predict_index + 1] = pred_word_id
          predict_index += 1

          probs, preds, state = sess.run(
              [ self.token_probs, self.predicted_tokens, self.next_state ],
              feed_dict={
                self.state : state,
                self.data_raw : source
                })
        
          pred_word_id = preds[0][predict_index]
          pred_word = self.id_tok[pred_word_id]
          prob_pred = probs[0][predict_index][pred_word_id] 
          log.debug('pred: %s' % pred_word)
          log.debug('prob: %f' % prob_pred)

      self.coord.request_stop()
      self.coord.join(self.threads)

  def force_decode(self, data):
    log.info('Force decoding test data...')
    verbose = FLAGS.output_mode == 'verbose'
    with tf.Session(graph=self.graph,
                    config=tf.ConfigProto(allow_soft_placement=verbose,
                    log_device_placement=verbose,
                    gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
      self._load(sess)

      for line in data:
        state = self.initial_state

        source = np.pad(np.array(line), 
            (0, self.batch_len - len(line) % self.batch_len), 'constant')
        predict_index = len(line) - 1

        target, score, state = sess.run(
            [ self.data_target, self.score, self.next_state ],
            feed_dict={
              self.state : state,
              self.data_raw : source
              })
        target.shape = (FLAGS.batch_size * FLAGS.max_time_steps,)
        log.debug('score: %.4f\ttarget: %s' 
            % (np.mean(score[:predict_index]) * 10000.0, 
               convert_id_tok([target[:predict_index]], self.id_tok)))

      self.coord.request_stop()
      self.coord.join(self.threads)


  def train(self):
    verbose = FLAGS.output_mode == 'verbose'
    with tf.Session(graph=self.graph,
                    config=tf.ConfigProto(allow_soft_placement=verbose,
                    log_device_placement=verbose,
                    gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
      self._load_or_create(sess)

      file_writer = tf.summary.FileWriter(
          os.path.join(FLAGS.tf_log_dir, FLAGS.run_name),
          sess.graph)
      summaries = tf.summary.merge_all()

      log.info('Starting training...')
      log.debug('Training %s' % tf.trainable_variables())

      cum_loss = 0.0
      for epoch in range(FLAGS.max_epochs):
        state = self.initial_state
        time_begin = timeit.default_timer()
        words_processed = 0

        for step in range(self.batches_per_epoch):
          batch_index = sess.run([ self.batch_select ])[0]
          data_begin = batch_index * self.batch_len
          data_end = data_begin + self.batch_len
          words_processed += FLAGS.batch_size * FLAGS.max_time_steps

          global_step = tf.train.global_step(sess, self.global_step)

          _, loss, summary_output, state, source, target, out = sess.run(
              [ self.minimizer,  self.loss, summaries,
                self.next_state,
                self.data_input,
                self.data_target,
                self.predicted_tokens ],
              feed_dict={
                self.state : state,
                self.data_raw : self.train_data[data_begin : data_end]
              })

          cum_loss = loss + cum_loss
          if global_step % 1000 == 0:
            time_elapsed = timeit.default_timer() - time_begin
            wps = float(words_processed) / time_elapsed 

            log.info('Epoch: %d, step: %d, wps: %.2f, loss %s'
                % (epoch, global_step, wps, cum_loss))
            log.debug('\n\tsrc: %s\n\ttarg: %s\n\tpred: %s' 
                % (convert_id_tok(source, self.id_tok)[:FLAGS.max_time_steps],
                   convert_id_tok(target, self.id_tok)[:FLAGS.max_time_steps],
                   convert_id_tok(out, self.id_tok)[:FLAGS.max_time_steps]))

            log.debug('Saved model checkpoint to %s.' % FLAGS.checkpoint_prefix)
            self.saver.save(sess, FLAGS.checkpoint_prefix, 
                global_step=global_step)

            cum_loss = 0
            words_processed = 0
            time_begin = timeit.default_timer()

          if global_step % 5000 == 0:
            self.validate(sess)

          file_writer.add_summary(summary_output, 
              global_step=global_step)

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
  #model.train()

  #predict_data, _, _ = reader.prepare_data(FLAGS.predict_data,
  #    FLAGS.vocab_data, FLAGS.vocab_size)
  #model.predict(predict_data)

  score_data , _, _ = reader.prepare_data(FLAGS.score_data,
      FLAGS.vocab_data, FLAGS.vocab_size)
  model.force_decode(score_data)

if __name__ == '__main__':
  if FLAGS.output_mode == 'debug' or FLAGS.output_mode == 'verbose':
    log.getLogger().setLevel(log.DEBUG)
  elif FLAGS.output_mode == 'info':
    log.getLogger().setLevel(log.INFO)
  tf.app.run()

 
