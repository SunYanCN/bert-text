import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from bert_text.layers import BertLayer
K.set_learning_phase(0)

bert_path = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
model_path = "/home/CAIL/bert_text/examples/Classify/saved_models/mdoel.h5"
model = load_model(model_path,custom_objects=BertLayer().get_custom_objects())

legacy_init_op = tf.group(tf.tables_initializer())

with K.get_session() as sess:
    export_path = 'saved_models'
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    signature_inputs = {
        'input_ids': tf.saved_model.utils.build_tensor_info(model.input[0]),
        'input_masks': tf.saved_model.utils.build_tensor_info(model.input[1]),
        'segment_ids': tf.saved_model.utils.build_tensor_info(model.input[2])
    }

    signature_outputs = {
        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: tf.saved_model.utils.build_tensor_info(
            model.output)
    }

    classification_signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=signature_inputs,
        outputs=signature_outputs,
        method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict': classification_signature_def
        },
        legacy_init_op=legacy_init_op
    )

    builder.save()