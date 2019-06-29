from bert_text.dataset import LCQMC
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

    dataset = LCQMC()
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
Converting examples to features: 100%|█| 238766/238766 [00:52<00:00, 4570.24it/s]
Converting examples to features: 100%|████| 8802/8802 [00:02<00:00, 4399.06it/s]

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          (None, 100)          0                                            
__________________________________________________________________________________________________
input_masks (InputLayer)        (None, 100)          0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        (None, 100)          0                                            
__________________________________________________________________________________________________
bert_layer_1 (BertLayer)        [(None, 100, 768), ( 102880904   input_ids[0][0]                  
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


Epoch 1/4 3363s 14ms/sample - loss: 0.3100 - acc: 0.8671 - val_loss: 0.4897 - val_acc: 0.7660
Epoch 2/4 4248s 18ms/sample - loss: 0.2745 - acc: 0.8841 - val_loss: 0.4944 - val_acc: 0.7746
Epoch 3/4 4250s 18ms/sample - loss: 0.2604 - acc: 0.8911 - val_loss: 0.4988 - val_acc: 0.7726
Epoch 4/4 4246s 18ms/sample - loss: 0.2503 - acc: 0.8957 - val_loss: 0.4857 - val_acc: 0.7802
"""