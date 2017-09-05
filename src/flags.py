import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 1.3, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.9, 'Initial momentum.')
flags.DEFINE_float('prob_dropout', 0.5, 'Dropout probability.')
flags.DEFINE_integer('max_epochs', 10, 'Maximum number of epochs.')
flags.DEFINE_integer('hidden_dim', 196, 'RNN hidden state size.')
flags.DEFINE_integer('max_time_steps', 30, 'Truncated backprop length.')
flags.DEFINE_integer('batch_size', 64, 'Num examples per minibatch.')
flags.DEFINE_integer('hidden_layers', 2, 'Num of RNN layers.')
flags.DEFINE_integer('max_grad_norm', 10, 'Clip gradients above this norm.')

flags.DEFINE_integer('vocab_size', 40000, 'Vocabulary size.')
flags.DEFINE_string('vocab_data', '../data/vocab.pkl', 'Vocabulary file.')
flags.DEFINE_string('train_data', '../data/train.shuf.txt', 'Training data.')
flags.DEFINE_string('dev_data', '../data/dev.txt', 'Validation data.')
flags.DEFINE_string('predict_data', '../data/test.txt',
    'Generate sentences for each seed line of this file.')

flags.DEFINE_string('checkpoint_prefix', 
  '/nfs/topaz/lcheung/models/tf-test/model',
  'Prefix of checkpoint files.')
flags.DEFINE_string('run_name', 
  'testrun',
  'Run name in tensorboard.')
flags.DEFINE_integer('valid_freq', 5000, 
  'Run validation test every this amount of steps.')
flags.DEFINE_integer('save_freq', 2000, 
  'Save a model checkpoint every this amount of steps.')

flags.DEFINE_string('output_mode', 'debug', 'verbose|debug|info')
flags.DEFINE_string('tf_log_dir', '/nfs/topaz/lcheung/tensorboard',
  'Path to store tensorboard log files.')

FLAGS = flags.FLAGS

