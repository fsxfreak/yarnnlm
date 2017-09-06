'''
Outputs likely sentences with rare words substituted into the sentence.
'''
import logging as log
import os, sys

import tensorflow as tf
import numpy as np

import reader, util
from flags import * # for FLAGS
from model import RNNLM

log.basicConfig(stream=sys.stderr, level=log.INFO,
    format='%(asctime)s [%(levelname)s]:%(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S')                        

flags.DEFINE_string('rare_src', '../data/rare.txt',
    'List of words deemed as rare in the parallel training data.')
flags.DEFINE_string('train_src', '../data/train.src',
    'Original training data source.')
flags.DEFINE_string('out_src', '../data/train.src.fwd.rare-aug',
    'Source sentences with a proposed rare augmentation.')

def main(_):
  # train_data and dev_data should be unused. using this for the vocab
  train_data, dev_data, tok_id, id_tok = reader.get_train_data(
      FLAGS.train_data, FLAGS.dev_data, FLAGS.vocab_data, FLAGS.vocab_size)

  train_src, _, _ = reader.prepare_data(FLAGS.train_src, 
      FLAGS.vocab_data, FLAGS.vocab_size)
  rare_src, _, _ = reader.prepare_data(FLAGS.rare_src,
      FLAGS.vocab_data, FLAGS.vocab_size)
  rare_src = reader.squash_data(rare_src)

  log.debug('train_src: %s' % train_src[:3])
  log.debug('rare_src: %s' % rare_src[:3])

  model = RNNLM(FLAGS, train_data, dev_data, id_tok)
  index_subs = model.find_rare_subs(train_src, rare_src)

  log.debug('Got %d substitutions.' % len(index_subs))
  log.info('Writing results to %s.' % FLAGS.out_src)
  with open(FLAGS.out_src, 'w') as f:
    for index, sub in subs:
      humanized = util.convert_id_tok([ sub ], id_tok)
      f.write('%d\t%s\n' % (index, humanized))

if __name__ == '__main__':
  if FLAGS.output_mode == 'debug' or FLAGS.output_mode == 'verbose':
    log.getLogger().setLevel(log.DEBUG)
  elif FLAGS.output_mode == 'info':
    log.getLogger().setLevel(log.INFO)
  tf.app.run()
