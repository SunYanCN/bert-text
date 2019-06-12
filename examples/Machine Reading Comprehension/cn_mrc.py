from bert_text.dataset import MSRA_NER
from bert_text.prepare.classification import pipline
import tensorflow as tf
from bert_text.layers import BertLayer
from tensorflow.keras.layers import Bidirectional, LSTM
from bert_text.layers import BertSquadLogitsLayer

def build_model(bert_path, max_seq_length, num_labels):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(bert_path=bert_path, n_fine_tune_layers=3, pooling="mean")(bert_inputs)
    squad_logits_layer = BertSquadLogitsLayer(name='squad_logits')
    start_logits, end_logits = squad_logits_layer(bert_output)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=[in_id, start_logits, end_logits])
    model.compile(loss=CRF.loss, optimizer="adam", metrics=CRF.viterbi_accuracy)
    model.summary()

    return model


if __name__ == '__main__':
    dataset = MSRA_NER()
    bert_path = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
    max_seq_length = 256

    train_inputs, train_labels, dev_inputs, dev_labels = pipline(bert_path=bert_path,
                                                                 dataset=dataset,
                                                                 max_seq_length=max_seq_length)

    model = build_model(bert_path, max_seq_length, num_labels=dataset.num_labels())

    model.fit(train_inputs, train_labels,
              validation_data=(dev_inputs, dev_labels),
              epochs=1, batch_size=10)
