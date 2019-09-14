import numpy as np
import tensorflow as tf
from tqdm import tqdm


def convert_string_to_bert_input(tokenizer, input_string, max_seq_length=128):
    tokens = []
    tokens.append("[CLS]")
    tokens.extend(tokenizer.tokenize(input_string))
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0: (max_seq_length - 2)]
    tokens.append("[SEP]")

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(tokens)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return tokens, input_ids, input_mask, segment_ids


def convert_string_to_bert_input_for_bert_module(tokenizer, input_string, max_seq_length=128):
    tokens, input_ids, input_mask, segment_ids = convert_string_to_bert_input(tokenizer, input_string, max_seq_length)
    return tokens, np.array([input_ids]), np.array([input_mask]), np.array([segment_ids])


def convert_string_list_to_features(tokenizer, string_list, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    token, input_ids, input_masks, segment_ids = [], [], [], []
    for example in tqdm(string_list, desc="Converting examples to features"):
        tokens, input_id, input_mask, segment_id = convert_string_to_bert_input(tokenizer, example, max_seq_length)
        token.append(tokens)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
    return [token,
            np.array(input_ids, dtype=np.int32),
            np.array(input_masks, dtype=np.int32),
            np.array(segment_ids, dtype=np.int32)]


def sentence_embedding(model, tokenizer, input_string, max_seq_length=128):
    token, input_ids, input_mask, segment_ids = convert_string_to_bert_input_for_bert_module(tokenizer,
                                                                                             max_seq_length=max_seq_length,
                                                                                             input_string=input_string)
    bert_inputs = dict(input_ids=input_ids, input_masks=input_mask, segment_ids=segment_ids)

    pooled_output, sequence_output = model.predict(bert_inputs, steps=1)

    return token, pooled_output, sequence_output


def tfdata_generator(inputs, batch_size=None):
    '''Construct a data generator using tf.Dataset'''
    input_ids, input_masks, segment_ids = inputs

    dataset = tf.data.Dataset.from_tensor_slices(
        dict(input_ids=input_ids, input_masks=input_masks, segment_ids=segment_ids), )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def sentence_embedding_batch(model, tokenizer, string_list, max_seq_length=128, batch_size=128):
    token, input_ids, input_mask, segment_ids = convert_string_list_to_features(tokenizer,
                                                                                max_seq_length=max_seq_length,
                                                                                string_list=string_list)
    dataset = tfdata_generator([input_ids, input_mask, segment_ids], batch_size=batch_size)
    dataset = dataset.make_one_shot_iterator()
    pooled_output, sequence_output = model.predict(dataset, steps=1)

    return token, pooled_output, sequence_output
