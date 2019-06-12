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

    bert_output = BertLayer(bert_path=bert_path, n_fine_tune_layers=0)(bert_inputs)
    dense = tf.keras.layers.Dense(128, activation="relu")(bert_output)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model

def initialize_vars(allow_growth=True):
    gpu_options = tf.GPUOptions(allow_growth=allow_growth)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

if __name__ == '__main__':
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
              epochs=4,batch_size=24,shuffle=True )

import time
saved_model_path = "./saved_models/{}".format(int(time.time()))
tf.keras.experimental.export_saved_model(model, saved_model_path)

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

Epoch 1/4  loss: 0.3052 - acc: 0.8754 - val_loss: 0.2681 - val_acc: 0.9058
Epoch 2/4  loss: 0.2318 - acc: 0.9099 - val_loss: 0.2462 - val_acc: 0.9167
Epoch 3/4  loss: 0.2089 - acc: 0.9195 - val_loss: 0.2325 - val_acc: 0.9225
Epoch 4/4  loss: 0.1951 - acc: 0.9220 - val_loss: 0.2336 - val_acc: 0.9192
"""
