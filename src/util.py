def convert_id_tok(batches, id_tok):
  '''
  batches: list of sentences (which are lists of ids)
  id_tok: dictionary associating id to readable token

  return: space delimited string of tokens
  '''
  s_raw = [ id_tok[sample] for batch in batches for sample in batch 
              if sample < len(id_tok)]
  return ' '.join(s_raw)
