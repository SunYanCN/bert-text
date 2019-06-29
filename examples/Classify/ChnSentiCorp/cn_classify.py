from bert_text.dataset import ChnSentiCorp
from bert_text.prepare.classification import pipline
import tensorflow as tf
from bert_text.layers import BertLayer
from tensorflow.keras import backend as K
from bert_text.utils import save_h5_model

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

def initialize_vars(allow_growth=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    sess = tf.Session(config=config)
    K.set_session(sess)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

if __name__ == '__main__':

    from  bert_text.utils.system import monitor_system_and_save,plot_result
    sys_info_filename = 'data.csv'
    monitor_system_and_save(filename=sys_info_filename)

    dataset = ChnSentiCorp()
    bert_path = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
    max_seq_length = 256

    train_inputs, train_labels, dev_inputs, dev_labels = pipline(bert_path=bert_path,
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
Converting examples to features: 100%|████| 9600/9600 [00:07<00:00, 1236.69it/s]
Converting examples to features: 100%|████| 1200/1200 [00:00<00:00, 1366.45it/s]
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          (None, 256)          0                                            
__________________________________________________________________________________________________
input_masks (InputLayer)        (None, 256)          0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        (None, 256)          0                                            
__________________________________________________________________________________________________
bert_layer_1 (BertLayer)        (None, 768)          102880904   input_ids[0][0]                  
                                                                 input_masks[0][0]                
                                                                 segment_ids[0][0]                
__________________________________________________________________________________________________
dense (Dense)                   (None, 128)          98432       bert_layer_1[0][0]               
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            129         dense[0][0]                      
==================================================================================================
Total params: 102,979,465
Trainable params: 98,561
Non-trainable params: 102,880,904
__________________________________________________________________________________________________

Epoch 1/4  248s 26ms/sample - loss: 0.3571 - acc: 0.8547 - val_loss: 0.3171 - val_acc: 0.8800
Epoch 2/4  248s 26ms/sample - loss: 0.2582 - acc: 0.9035 - val_loss: 0.2856 - val_acc: 0.8925
Epoch 3/4  247s 26ms/sample - loss: 0.2386 - acc: 0.9094 - val_loss: 0.2604 - val_acc: 0.9083
Epoch 4/4  247s 26ms/sample - loss: 0.2167 - acc: 0.9193 - val_loss: 0.2658 - val_acc: 0.9067

mean mem_used:3.565377434914118,mean_gpu_0_proc:97.24629812438302
"""
