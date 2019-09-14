import collections

import numpy as np
import tensorflow as tf
from bert.run_squad import _improve_answer_span, _check_is_max_context
from tqdm import tqdm

from .utils import ChineseFullTokenizer


def convert_single_example(tokenizer, example, is_training, max_query_length, max_seq_length, doc_stride):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        if is_training:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        return input_ids, input_mask, segment_ids, start_position, end_position


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    unique_ids, example_indexs, input_ids, input_masks, segment_ids, start_positions, end_positions = [], [], [], [], [], [], []

    for (example_index, example) in tqdm(enumerate(examples), desc="Converting examples to feature"):
        feature = convert_single_example(example=example, tokenizer=tokenizer,
                                         is_training=is_training,
                                         doc_stride=doc_stride,
                                         max_query_length=max_query_length,
                                         max_seq_length=max_seq_length)
        input_id, input_mask, segment_id, start_position, end_position = feature

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        start_positions.append(start_position)
        end_positions.append(end_position)
        unique_ids.append(unique_id)

        unique_id += 1

    if is_training:
        features = [unique_ids, input_ids, input_masks, segment_ids, start_positions, end_positions]
    else:
        features = [unique_ids, input_ids, input_masks, segment_ids]

    features = [np.array(i) for i in features]
    return features


def tfdata_generator(inputs, is_training, max_seq_length, batch_size=None, epochs=None):
    '''Construct a data generator using tf.Dataset'''
    unique_ids, input_ids, input_masks, segment_ids, start_positions, end_positions = inputs

    def generator():
        for s0, s1, s2, s3, l1, l2 in zip(unique_ids, input_ids, input_masks, segment_ids, start_positions,
                                          end_positions):
            yield {"unique_ids": s0, "input_ids": s1, "input_masks": s2, "segment_ids": s3}, \
                  {"start_logits": l1, "end_logits": l2}

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=({"unique_ids": (),
                                                             "input_ids": (max_seq_length),
                                                             "input_masks": (max_seq_length),
                                                             "segment_ids": (max_seq_length)},
                                                            {"start_logits": (), "end_logits": ()}),
                                             output_types=(
                                                 {"unique_ids": tf.int64, "input_ids": tf.int64,
                                                  "input_masks": tf.int64, "segment_ids": tf.int64},
                                                 {"start_logits": tf.int64, "end_logits": tf.int64}))
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.cache()
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True if is_training else False)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def reading_comprehension_pipline(dataset, vocab_file, do_lower_case, doc_stride, max_seq_length, max_query_length,
                                  batch_size, epochs, return_tf_dataset=False):
    tokenizer = ChineseFullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    train_features = convert_examples_to_features(examples=dataset.get_train_examples(),
                                                  tokenizer=tokenizer,
                                                  max_seq_length=max_seq_length,
                                                  doc_stride=doc_stride,
                                                  max_query_length=max_query_length,
                                                  is_training=True)

    dev_features = convert_examples_to_features(examples=dataset.get_dev_examples(),
                                                tokenizer=tokenizer,
                                                max_seq_length=max_seq_length,
                                                doc_stride=doc_stride,
                                                max_query_length=max_query_length,
                                                is_training=True)

    if return_tf_dataset:
        train_dataset = tfdata_generator(train_features,
                                         is_training=True,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         max_seq_length=max_seq_length)
        dev_dataset = tfdata_generator(dev_features,
                                       is_training=False,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       max_seq_length=max_seq_length)

        return train_dataset.make_one_shot_iterator(), dev_dataset.make_one_shot_iterator()

    else:
        train_inputs = train_features[:-2]
        train_labels = train_features[-2:]
        dev_inputs = dev_features[:-2]
        dev_labels = dev_features[-2:]

        return train_inputs, train_labels, dev_inputs, dev_labels
