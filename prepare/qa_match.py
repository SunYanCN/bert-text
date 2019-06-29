import tensorflow as tf
import numpy as np
from tqdm import tqdm
from bert.tokenization import FullTokenizer
from bert.run_classifier import _truncate_seq_pair

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

def qa_match_convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = tokenizer.tokenize(example.text_b)

    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    # if len(tokens_a) > max_seq_length - 2:
    #     tokens_a = tokens_a[0: (max_seq_length - 2)]
    #
    # if len(tokens_b) > max_seq_length - 2:
    #     tokens_b = tokens_b[0: (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for i, token in enumerate(tokens_a):
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
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

    return input_ids, input_mask, segment_ids, example.label


def qa_match_convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = qa_match_convert_single_example(
            tokenizer=tokenizer, example=example, max_seq_length=max_seq_length)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return [
        np.asarray(input_ids),
        np.asarray(input_masks),
        np.asarray(segment_ids),
        np.asarray(labels)]

def qa_match_pipline(vocab_file, dataset, max_seq_length):

    tokenizer = FullTokenizer(vocab_file=vocab_file)
    train_input_ids, train_input_masks, train_segment_ids, train_labels = qa_match_convert_examples_to_features(tokenizer,
                                                                                                       dataset.get_train_examples(),
                                                                                                       max_seq_length=max_seq_length)
    dev_input_ids, dev_input_masks, dev_segment_ids, dev_labels = qa_match_convert_examples_to_features(tokenizer,
                                                                                               dataset.get_dev_examples(),
                                                                                               max_seq_length=max_seq_length)

    train_inputs = [train_input_ids, train_input_masks, train_segment_ids]
    dev_inputs = [dev_input_ids, dev_input_masks, dev_segment_ids]

    return train_inputs, train_labels, dev_inputs, dev_labels