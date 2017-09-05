'''
Projects sentences into the encoded space and saves those vectors to a .npy
file.
'''
import logging as log
import os, sys

import tensorflow as tf
import numpy as np

import reader
from flags import * # for FLAGS
from model import RNNLM

log.basicConfig(stream=sys.stderr, level=log.INFO,
    format='%(asctime)s [%(levelname)s]:%(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S')                        

flags.DEFINE_string('vector_data', '../data/play.txt',
    'Generate vector representations for each of the lines in this file.')
flags.DEFINE_string('out_vectors', '../data/play.txt.vec',
    'Generate vector representations for each of the lines in this file.')

def main(_):
  train_data, dev_data, tok_id, id_tok = reader.get_train_data(
      FLAGS.train_data, FLAGS.dev_data, FLAGS.vocab_data, FLAGS.vocab_size)

  vector_data, _, _ = reader.prepare_data(FLAGS.vector_data,
      FLAGS.vocab_data, FLAGS.vocab_size)

  model = RNNLM(FLAGS, train_data, dev_data, id_tok)
  log.debug('Num lines: %d' % len(vector_data))
  vecs = model.find_vectors(vector_data)
  log.debug('Num vecs: %d' % len(vecs))

  log.debug('Writing vectors to %s' % FLAGS.out_vectors)
  np.save(FLAGS.out_vectors, vecs)

if __name__ == '__main__':
  if FLAGS.output_mode == 'debug' or FLAGS.output_mode == 'verbose':
    log.getLogger().setLevel(log.DEBUG)
  elif FLAGS.output_mode == 'info':
    log.getLogger().setLevel(log.INFO)
  tf.app.run()
