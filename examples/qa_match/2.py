from bert_text.dataset import NLPCC_DBQA
from bert_text.prepare.qa_match import qa_match_pipline
import tensorflow as tf
from bert_text.layers import BertLayer
from tensorflow.keras import backend as K
from bert_text.utils import save_h5_model
from bert_text.utils.init import initialize_vars

def build_model(bert_path, max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    sequence_output, pooled_output = BertLayer(bert_path=bert_path, trainable=False,
                                               n_fine_tune_layers=0,
                                               max_len=max_seq_length)(bert_inputs)
    dense = tf.keras.layers.Dense(128, activation="relu")(pooled_output)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model

if __name__ == '__main__':

    from  bert_text.utils.system import monitor_system_and_save,plot_result
    sys_info_filename = 'data.csv'
    monitor_system_and_save(filename=sys_info_filename)

    dataset = NLPCC_DBQA()
    bert_path = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
    vocab_file = "/home/CAIL/bert_text/examples/SequenceLabel/vocab.txt"
    max_seq_length = 256

    train_inputs, train_labels, dev_inputs, dev_labels = qa_match_pipline(vocab_file=vocab_file,
                                                                           dataset=dataset,
                                                                           max_seq_length=max_seq_length)

    model = build_model(bert_path, max_seq_length)

    initialize_vars()

    model.fit(train_inputs, train_labels,
              validation_data=(dev_inputs, dev_labels),
              epochs=4,batch_size=128,shuffle=True)

    plot_result(sys_info_filename)

    # save_h5_model(model)

# tf.keras.experimental.export_saved_model(model, saved_model_path)

"""
Converting examples to features: 100%|█| 181882/181882 [01:40<00:00, 1801.70it/s]
Converting examples to features: 100%|██| 40995/40995 [00:19<00:00, 2060.08it/s]
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          (None, 256)          0                                            
__________________________________________________________________________________________________
input_masks (InputLayer)        (None, 256)          0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        (None, 256)          0                                            
__________________________________________________________________________________________________
bert_layer_1 (BertLayer)        [(None, 256, 768), ( 102880904   input_ids[0][0]                  
                                                                 input_masks[0][0]                
                                                                 segment_ids[0][0]                
__________________________________________________________________________________________________
dense (Dense)                   (None, 128)          98432       bert_layer_1[0][1]               
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            129         dense[0][0]                      
==================================================================================================
Total params: 102,979,465
Trainable params: 98,561
Non-trainable params: 102,880,904
__________________________________________________________________________________________________

Epoch 1/4  10649s 59ms/sample - loss: 0.1597 - acc: 0.9498 - val_loss: 0.1330 - val_acc: 0.9546
Epoch 2/4  7551s 42ms/sample - loss: 0.1394 - acc: 0.9537 - val_loss: 0.1348 - val_acc: 0.9541
Epoch 3/4  5083s 28ms/sample - loss: 0.1306 - acc: 0.9561 - val_loss: 0.1225 - val_acc: 0.9583
Epoch 4/4  5093s 28ms/sample - loss: 0.1232 - acc: 0.9580 - val_loss: 0.1184 - val_acc: 0.9591
"""