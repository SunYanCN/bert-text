import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from .utils import ChineseFullTokenizer


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


def ner_convert_single_example(tokenizer, example, label_map=None, output_dir=".", max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    labels = example.label.split('\x02')

    label_map = label_map

    tokens_a = tokenizer.tokenize(example.text_a.replace('\x02', ''))

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]
        labels = labels[0: (max_seq_length - 2)]

    tokens = []
    label_ids = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["O"])

    for i, token in enumerate(tokens_a):
        tokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["O"])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, label_ids


def ner_convert_examples_to_features(tokenizer, examples, max_seq_length=256, label_map=None, output_dir="."):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = ner_convert_single_example(
            tokenizer=tokenizer, example=example, max_seq_length=max_seq_length, label_map=label_map,
            output_dir=output_dir)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return [
        np.asarray(input_ids),
        np.asarray(input_masks),
        np.asarray(segment_ids),
        np.asarray(labels)]


def tfdata_generator(inputs, labels, is_training, max_seq_length, batch_size=None, epochs=None):
    '''Construct a data generator using tf.Dataset'''
    input_ids, input_masks, segment_ids = inputs

    def generator():
        for s1, s2, s3, l in zip(input_ids, input_masks, segment_ids, labels):
            yield {"input_ids": s1, "input_masks": s2, "segment_ids": s3}, l

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=({"input_ids": (max_seq_length),
                                                             "input_masks": (max_seq_length),
                                                             "segment_ids": (max_seq_length)},
                                                            (1)),
                                             output_types=(
                                                 {"input_ids": tf.int64, "input_masks": tf.int64,
                                                  "segment_ids": tf.int64},
                                                 tf.int64))
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.cache()
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True if is_training else False)
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def ner_pipline(vocab_file, do_lower_case, dataset, max_seq_length, num_labels, label_map=None, output_dir="."):
    tokenizer = ChineseFullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    train_input_ids, train_input_masks, train_segment_ids, train_labels = ner_convert_examples_to_features(
        tokenizer=tokenizer,
        examples=dataset.get_train_examples(),
        max_seq_length=max_seq_length,
        label_map=label_map,
        output_dir=output_dir)
    dev_input_ids, dev_input_masks, dev_segment_ids, dev_labels = ner_convert_examples_to_features(tokenizer,
                                                                                                   dataset.get_dev_examples(),
                                                                                                   max_seq_length=max_seq_length,
                                                                                                   label_map=label_map,
                                                                                                   output_dir=output_dir)

    train_inputs = [train_input_ids, train_input_masks, train_segment_ids]
    dev_inputs = [dev_input_ids, dev_input_masks, dev_segment_ids]

    train_labels = to_categorical(train_labels, num_classes=num_labels)
    dev_labels = to_categorical(dev_labels, num_classes=num_labels)

    return train_inputs, train_labels, dev_inputs, dev_labels
