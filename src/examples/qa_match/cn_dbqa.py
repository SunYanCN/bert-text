from dataset import NLPCC_DBQA
from layers.bert import load_pretrained_model
from prepare.qa_match import qa_match_pipline
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_model(config_path, checkpoint_path, max_seq_length, bert_trainable=False):
    in_id = Input(shape=(max_seq_length,), name="input_ids", dtype="int32")
    in_segment = Input(shape=(max_seq_length,), name="segment_ids", dtype="int32")

    bert_model = load_pretrained_model(config_path, checkpoint_path)  # 建立模型，加载权重

    for l in bert_model.layers:
        if bert_trainable:
            l.trainable = True
        else:
            l.trainable = False

    sequence_output = bert_model([in_id, in_segment])
    pooled_output = Lambda(lambda x: x[:, 0])(sequence_output)
    pred = Dense(1, activation="sigmoid")(pooled_output)

    model = Model(inputs=[in_id, in_segment], outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
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
    batch_size = 256

    dataset = NLPCC_DBQA()
    dataset.print_info(5)

    train_inputs, train_labels, dev_inputs, dev_labels = qa_match_pipline(vocab_file=vocab_file,
                                                                          do_lower_case=False,
                                                                          dataset=dataset,
                                                                          max_seq_length=max_seq_length)

    model = build_model(config_path=config_path, checkpoint_path=checkpoint_path, max_seq_length=max_seq_length,
                        bert_trainable=False)

    model.fit(train_inputs, train_labels,
              validation_data=(dev_inputs, dev_labels),
              epochs=epochs, batch_size=batch_size, shuffle=True)

    # save_h5_model(model)

# tf.keras.experimental.export_saved_model(model, saved_model_path)
