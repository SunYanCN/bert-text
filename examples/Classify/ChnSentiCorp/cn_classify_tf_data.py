from bert_text.dataset import ChnSentiCorp
from bert_text.prepare.classification import pipline
import tensorflow as tf
from bert_text.layers import BertLayer
from tensorflow.keras import backend as K

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

    from bert_text.utils.system import monitor_system_and_save, plot_result
    import math

    sys_info_filename = 'tf_data.csv'
    monitor_system_and_save(filename=sys_info_filename)

    dataset = ChnSentiCorp()
    bert_path = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"

    max_seq_length = 256
    epochs =4
    batch_size = 128

    train_dataset, dev_dataset = pipline(bert_path=bert_path,
                                         dataset=dataset,
                                         batch_size = batch_size,
                                         epochs = epochs,
                                         max_seq_length=max_seq_length,
                                         return_tf_dataset=True)

    model = build_model(bert_path, max_seq_length)

    initialize_vars()

    model.fit(train_dataset,
              validation_data=dev_dataset, batch_size=batch_size,
              epochs=epochs, steps_per_epoch=dataset.train_num//128, validation_steps= math.ceil(dataset.dev_num/128))

    plot_result(sys_info_filename)

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

Epoch 1/4  248s 3s/step - loss: 0.3365 - acc: 0.8697 - val_loss: 0.2940 - val_acc: 0.8967
Epoch 2/4  246s 3s/step - loss: 0.2530 - acc: 0.9095 - val_loss: 0.2723 - val_acc: 0.8976
Epoch 3/4  246s 3s/step - loss: 0.2343 - acc: 0.9135 - val_loss: 0.2573 - val_acc: 0.9071
Epoch 4/4  246s 3s/step - loss: 0.2200 - acc: 0.9174 - val_loss: 0.2586 - val_acc: 0.9062

mean mem_used:3.6167192429642716,mean_gpu_0_proc:97.29809104258443
"""
