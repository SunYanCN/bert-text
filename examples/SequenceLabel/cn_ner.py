from bert_text.dataset import MSRA_NER
from bert_text.prepare.classification import ner_pipline
import tensorflow as tf
from bert_text.layers import BertLayer, CRF
from tensorflow.keras.layers import Bidirectional, LSTM, CuDNNLSTM
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.callbacks import Callback

from bert_text.callbacks import TqdmCallback

from seqeval.metrics import f1_score, accuracy_score, classification_report


class TestCallback(Callback):
    def __init__(self, XY, model, tags):
        self.X, self.Y = XY
        self.Y = np.argmax(self.Y, -1)
        self.smodel = model
        self.tags = tags
        self.best_f1 = 0

    def on_epoch_end(self,epoch, logs=None):
        # self.model is auto set by keras
        yt, yp = [], []
        pred = np.argmax(self.smodel.predict(self.X, batch_size=32), -1)
        lengths = [x.sum() for x in self.X[1]]
        for pseq, yseq, llen in zip(pred, self.Y, lengths):
            yt.append([self.tags[z] for z in pseq[1:llen - 1]])
            yp.append([self.tags[z] for z in yseq[1:llen - 1]])
        f1 = f1_score(yt, yp)
        self.best_f1 = max(self.best_f1, f1)
        accu = accuracy_score(yt, yp)
        print('\naccu: %.4f  F1: %.4f  BestF1: %.4f\n' % (accu, f1, self.best_f1))
        print(classification_report(yt, yp))


def build_model(bert_path, max_seq_length, label_num):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    sequence_output, pooled = BertLayer(bert_path=bert_path, trainable=False,
                                        n_fine_tune_layers=0, max_len=max_seq_length, name="Bert_layer")(bert_inputs)
    bilstm_output = Bidirectional(LSTM(128, return_sequences=True))(sequence_output)
    dense = tf.keras.layers.Dense(label_num, activation="relu")(bilstm_output)
    crf = CRF(label_num)
    pred = crf(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss=crf.loss, optimizer="adam", metrics=[crf.viterbi_accuracy])
    model.summary(line_length=150)

    return model


def initialize_vars(allow_growth=True):
    gpu_options = tf.GPUOptions(allow_growth=allow_growth)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


if __name__ == '__main__':
    dataset = MSRA_NER()
    bert_path = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
    max_seq_length = 256

    train_inputs, train_labels, dev_inputs, dev_labels = ner_pipline(bert_path=bert_path,
                                                                     dataset=dataset,
                                                                     max_seq_length=max_seq_length,
                                                                     label_map=dataset.label_map,
                                                                     num_labels=dataset.num_labels)

    model = build_model(bert_path, max_seq_length, label_num=dataset.num_labels)

    test_cb = TestCallback((dev_inputs, dev_labels), model, dataset.map_label)
    initialize_vars()

    # train_dataset_inputs = tf.data.Dataset.from_tensor_slices(train_inputs[0])
    # train_dataset_labels = tf.data.Dataset.from_tensor_slices(train_labels)
    # train_dataset = tf.data.Dataset.zip((train_dataset_inputs, train_dataset_labels)).batch(32).repeat()
    #
    # val_dataset_inputs = tf.data.Dataset.from_tensor_slices(dev_inputs)
    # val_dataset_labels = tf.data.Dataset.from_tensor_slices(dev_labels)
    # val_dataset = tf.data.Dataset.zip((val_dataset_inputs, val_dataset_labels)).batch(32).repeat()
    #
    # model.fit(train_dataset, epochs=4, steps_per_epoch=300,
    #           validation_data=val_dataset, validation_steps=50)

    model.fit(train_inputs, train_labels,
              validation_data=(dev_inputs, dev_labels),
              epochs=3, batch_size=128, verbose=1, callbacks=[test_cb])

"""
LSTM Time：
Epoch 1/3
69s 37ms/sample - loss: 20.4620 - accuracy: 0.9760 - val_loss: 261.6392 - val_accuracy: 0.9874
Epoch 2/3
 765s 37ms/sample - loss: 7.8265 - accuracy: 0.9895 - val_loss: 259.8727 - val_accuracy: 0.9903
Epoch 3/3
765s 37ms/sample - loss: 5.9192 - accuracy: 0.9922 - val_loss: 258.1928 - val_accuracy: 0.9935

866s 41ms/sample - loss: 11.5048 - accuracy: 0.9829 - val_loss: 188.1643 - val_accuracy: 0.9828
"""
