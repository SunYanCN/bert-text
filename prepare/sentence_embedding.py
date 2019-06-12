import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer


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

    return input_ids, input_mask, segment_ids

def convert_string_to_bert_input_for_bert_module(tokenizer, input_string, max_seq_length=128):
    input_ids, input_mask, segment_ids = convert_string_to_bert_input(tokenizer, input_string, max_seq_length)
    return np.array([input_ids]), np.array([input_mask]), np.array([segment_ids])

def sentence_embedding(bert_path, input_string, max_seq_length = 128):

    token = tokenizer.tokenize(input_string)

    bert_module = hub.Module(bert_path)

    input_ids, input_mask, segment_ids = convert_string_to_bert_input_for_bert_module(tokenizer, max_seq_length=max_seq_length,
                                                                                      input_string=input_string)
    bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)
    pooled_output = bert_outputs["pooled_output"]
    sequence_output = bert_outputs["sequence_output"]

    return token, pooled_output, sequence_output

if __name__ == '__main__':
    texts = ["我爱中国", "我爱学习"]
    bert_path = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
    tokenizer = create_tokenizer_from_hub_module(bert_path)

    for text in texts:

       token, pooled_output, sequence_output = sentence_embedding(bert_path= bert_path, max_seq_length=8,
                                                                  input_string=text)

       print(token)
       print(pooled_output.shape)
       print(sequence_output.shape)

