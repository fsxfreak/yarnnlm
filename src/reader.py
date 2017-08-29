from collections import OrderedDict

import logging as log
import os, sys 

import cPickle as pkl
import numpy as np

log.basicConfig(stream=sys.stderr, level=log.INFO)

PAD = '<PAD>'
UNK = '<UNK>'
EOS = '<EOS>'
PAD_ID = 0
UNK_ID = 1
EOS_ID = 2

def read_data(filename):
  '''
  read_data:  reads data from a line-delimited sentence file.

  filename:   path to file
  return:     python list of lists, each list in the list being a list of
              tokens for a sentence.
  '''

  lines = []
  with open(filename, 'r') as f:
    for line in f:
      line = line.strip()
      toks = line.split(' ')

      lines.append(toks)

  log.info('Finished reading data from %s.' % filename)
  return lines

def extract_reverse_vocab(tok_id):
  '''
  extract_reverse_vocab:  takes token to id mappings and returns id to
                          token mappings

  tok_id:                 OrderedDict of tok to id

  return:                 id_tok: OrderedDict of id to tok
  '''
  id_tok = OrderedDict()
  for k, v in tok_id.items():
    id_tok[v] = k
  return id_tok

def build_vocab(lines, max_vocab_size):
  '''
  build_vocab:      maps tokens to ids, starts at 3. 0 reserved for PAD,
                    1 reserved for EOS, 2 reserved for UNK 

  lines:            list of lists, as returned by read_data, each element a
                    list of tokens
  max_vocab_size:   maximum amount of tokens in vocabulary

  return:           tok_id: token to id dictionary
  '''
  counts = {}
  for toks in lines:
    for tok in toks:
      if tok not in counts:
        counts[tok] = 0
      counts[tok] += 1

  counts_sorted = sorted(counts.items(), key=lambda e: e[1], reverse=True)

  tok_id = OrderedDict()
  tok_id[PAD] = PAD_ID
  tok_id[UNK] = UNK_ID
  tok_id[EOS] = EOS_ID

  offset = len(tok_id)

  for i, (tok, _) in enumerate(counts_sorted):
    if len(tok_id) >= max_vocab_size:
      break
  
    assert tok not in tok_id  # unexpected repeat token
    tok_id[tok] = i + offset

  assert len(tok_id) <= max_vocab_size

  log.info('Finished building vocabulary from data.')
  return tok_id

def vocab_replace_data(lines, tok_id):
  '''
  vocab_replace_data: after vocabulary is built, convert the data to token ids.

  lines:              list of lists as returned by read_data
  tok_id:             OrderedDict mapping string to int (token to id)

  return:             lines: lines with UNK and EOS
  '''
  for toks in lines:
    for i, tok in enumerate(toks):
      if tok in tok_id:
        toks[i] = tok_id[tok]
      else:
        toks[i] = tok_id[UNK]
    toks.append(tok_id[EOS])

  log.info('Finished converting tokens to ids in data.')
  return lines

def save_vocab(filename, tok_id):
  '''
  save_vocab: save tok_id to filename, should be .pkl file
  '''
  with open(filename, 'wb') as f:
    pkl.dump(tok_id, f)
  log.debug('Saved vocab to %s.' % filename)

def load_vocab(vocab_filename, lines=None, max_vocab_size=None):
  '''
  load_vocab:     load or build vocab.

  vocab_filename: filename to load or save vocab to
  lines:          data to build vocab from
  max_vocab_size: self explanatory

  return:         id_tok: ids to tokens
                  tok_id: tokens to ids
  '''
  if os.path.exists(vocab_filename):
    with open(vocab_filename, 'rb') as f:
      tok_id = pkl.load(f)
    log.info('Loaded vocabulary of size: %d from %s.' 
        % (len(tok_id), vocab_filename))
  else:
    if lines is None or max_vocab_size is None:
      raise ValueError('Could not load vocabulary or create it.')
    tok_id = build_vocab(lines, max_vocab_size)
    log.info('Saved new vocabulary of size: %d to %s.' 
        % (len(tok_id), vocab_filename))
    save_vocab(vocab_filename, tok_id)

  if max_vocab_size:
    assert len(tok_id) <= max_vocab_size

  id_tok = extract_reverse_vocab(tok_id)

  return tok_id, id_tok

def squash_data(data):
  '''
  Squash all sentences together.
  '''
  squashed = np.concatenate(data)
  return squashed

def prepare_data(data_filename, vocab_filename, vocab_size=30000):
  '''
  prepare_data: prepares data from filenames to be fed into tensorflow

  return: data, tok_id, id_tok: np array, dictionary, dictionary
  '''
  raw_data = read_data(data_filename)
  tok_id, id_tok = load_vocab(vocab_filename, raw_data, vocab_size)

  data = np.array(vocab_replace_data(raw_data, tok_id))
  return data, tok_id, id_tok

def main():
  train_data = read_data('train.txt')

  tok_id, id_tok = load_vocab('vocab.pkl', train_data, 30000)

  train_data = vocab_replace_data(train_data, tok_id)
  log.debug(train_data[:5])

if __name__ == '__main__':
  log.getLogger().setLevel(log.DEBUG)
  main()

