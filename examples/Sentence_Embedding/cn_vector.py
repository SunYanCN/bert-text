from bert.tokenization import FullTokenizer
from bert_text.prepare.sentence_embedding import sentence_embedding
import tensorflow as tf
from bert_text.layers import BertLayer
from bert_text.utils.init import initialize_vars
from bert_text.prepare.sentence_embedding import sentence_embedding_batch

texts = ["我爱中国", "我爱学习"]
bert_path = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
vocab_file = "/home/CAIL/bert_text/examples/SequenceLabel/vocab.txt"
tokenizer = FullTokenizer(vocab_file=vocab_file)

def build_model(bert_path, max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    sequence_output, pooled_output = BertLayer(bert_path=bert_path, trainable=False,
                                               n_fine_tune_layers=0,
                                               max_len=max_seq_length)(bert_inputs)
    model = tf.keras.models.Model(inputs=bert_inputs, outputs=[sequence_output, pooled_output])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


model = build_model(bert_path, max_seq_length =8)

initialize_vars()

for text in texts:
    token, pooled_output, sequence_output = sentence_embedding(model,
                                                               tokenizer,
                                                               max_seq_length=8,
                                                               input_string=text)

    print(token)
    print(pooled_output.shape)
    print(sequence_output.shape)



token, pooled_output, sequence_output = sentence_embedding_batch(model=model,
                                                          tokenizer=tokenizer,
                                                          max_seq_length=8,
                                                          batch_size= 2,
                                                          string_list=texts)

print(token)
print(pooled_output.shape)
print(sequence_output.shape)
"""
['我', '爱', '中', '国']
(1, 768)
(1, 8, 768)
['我', '爱', '学', '习']
(1, 768)
(1, 8, 768)
"""

