import logging as log
import os, sys

import tensorflow as tf

import reader
from flags import * # for FLAGS
from model import RNNLM

log.basicConfig(stream=sys.stderr, level=log.INFO,
    format='%(asctime)s [%(levelname)s]:%(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S')                        

def main(_):
  train_data, dev_data, tok_id, id_tok = reader.get_train_data(
      FLAGS.train_data, FLAGS.dev_data, FLAGS.vocab_data, FLAGS.vocab_size)

  score_data , _, _ = reader.prepare_data(FLAGS.score_data,
      FLAGS.vocab_data, FLAGS.vocab_size)

  model = RNNLM(FLAGS, train_data, dev_data, id_tok)
  scores = model.force_decode(score_data)

  output_path = os.path.join(
    os.path.dirname(FLAGS.score_data),
    '%s.scores' % os.path.basename(FLAGS.score_data))
  log.debug('Writing scores to %s' % output_path)
  with open(output_path, 'w') as f:
    for score in scores:
      f.write('%f\n' % score)

if __name__ == '__main__':
  if FLAGS.output_mode == 'debug' or FLAGS.output_mode == 'verbose':
    log.getLogger().setLevel(log.DEBUG)
  elif FLAGS.output_mode == 'info':
    log.getLogger().setLevel(log.INFO)
  tf.app.run()
