import tensorflow as tf
import tensorflow.keras.backend as K


def initialize_vars(allow_growth=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    sess = tf.Session(config=config)
    K.set_session(sess)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
