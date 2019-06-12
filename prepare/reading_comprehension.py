from bert.tokenization import FullTokenizer
from tqdm import tqdm
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf


def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    sess = tf.Session()
    vocab_file, do_lower_case = sess.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )
    sess.close()
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_examples_to_features(examples):
    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
