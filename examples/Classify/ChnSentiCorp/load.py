import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from bert.tokenization import FullTokenizer

export_dir = '/home/CAIL/bert_text/examples/SequenceLabel/saved_models'

vocab_file = "/home/CAIL/bert_text/examples/SequenceLabel/vocab.txt"


text = "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"

tokenizer = FullTokenizer(vocab_file=vocab_file)

tokens = tokenizer.tokenize(text=text)

max_seq_length = 256

input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_masks = [1] * len(input_ids)
if len(input_ids) < max_seq_length:
    input_ids += [0] * (max_seq_length - len(input_ids))
    input_masks += [0] * (max_seq_length - len(input_masks))

segment_ids = [0] * max_seq_length

with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)
    signature = meta_graph_def.signature_def

    x1_tensor_name = signature['predict'].inputs['input_ids'].name
    x2_tensor_name = signature['predict'].inputs['input_masks'].name
    x3_tensor_name = signature['predict'].inputs['segment_ids'].name

    y_tensor_name = signature['predict'].outputs[
        signature_constants.CLASSIFY_OUTPUT_CLASSES].name
    x1 = sess.graph.get_tensor_by_name(x1_tensor_name)
    x2 = sess.graph.get_tensor_by_name(x2_tensor_name)
    x3 = sess.graph.get_tensor_by_name(x3_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)
    result = sess.run(y, feed_dict={x1: [input_ids], x2: [input_masks], x3: [segment_ids]})  # 预测值
    print(result.argmax(-1))