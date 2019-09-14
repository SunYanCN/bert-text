from dataset import MSRA_NER
from layers import CRF
from layers.bert import load_pretrained_model
from prepare.sequence_label import ner_pipline
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_model(config_path, checkpoint_path, max_seq_length, label_num, bert_trainable=False):
    in_id = Input(shape=(max_seq_length,), name="input_ids", dtype="int32")
    in_segment = Input(shape=(max_seq_length,), name="segment_ids", dtype="int32")

    bert_model = load_pretrained_model(config_path, checkpoint_path)  # 建立模型，加载权重

    for l in bert_model.layers:
        if bert_trainable:
            l.trainable = True
        else:
            l.trainable = False

    sequence_output = bert_model([in_id, in_segment])
    bilstm_output = Bidirectional(CuDNNLSTM(128, return_sequences=True))(sequence_output)

    layer_dense = Dense(64, activation='tanh', name='layer_dense')
    layer_crf_dense = Dense(label_num, name='layer_crf_dense')
    layer_crf = CRF(label_num, name='layer_crf')

    dense = layer_dense(bilstm_output)
    dense = layer_crf_dense(dense)
    pred = layer_crf(dense)

    model = Model(inputs=[in_id, in_segment], outputs=pred)
    model.compile(loss=layer_crf.loss, optimizer=Adam(lr=1e-5), metrics=[layer_crf.viterbi_accuracy])

    model.summary(line_length=150)

    return model


if __name__ == '__main__':
    import tensorflow as tf
    import tensorflow.keras.backend as K

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    config_path = "/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12/bert_config.json"
    checkpoint_path = "/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12/bert_model.ckpt"
    vocab_file = "/home/new/Toxicity/bert_model/models/chinese_L-12_H-768_A-12/vocab.txt"

    max_seq_length = 128
    epochs = 10
    batch_size = 10

    dataset = MSRA_NER()

    dataset.print_info(2)

    train_inputs, train_labels, dev_inputs, dev_labels = ner_pipline(vocab_file=vocab_file,
                                                                     do_lower_case=False,
                                                                     dataset=dataset,
                                                                     max_seq_length=max_seq_length,
                                                                     label_map=dataset.label_map,
                                                                     num_labels=dataset.num_labels)

    model = build_model(config_path=config_path, checkpoint_path=checkpoint_path,
                        max_seq_length=max_seq_length, bert_trainable=True,
                        label_num=dataset.num_labels)

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
              epochs=epochs, batch_size=batch_size)

    # save_h5_model(model)
