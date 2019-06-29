import tensorflow as tf

def get_bert_model(input_word_ids,
                   input_mask,
                   input_type_ids,
                   config=None,
                   name=None,
                   float_type=tf.float32):

  bert_model_layer = BertModel(config=config, float_type=float_type, name=name)
  pooled_output, sequence_output = bert_model_layer(input_word_ids, input_mask,
                                                    input_type_ids)
  bert_model = tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output])
  return bert_model

core_model = get_bert_model(
      input_word_ids,
      input_mask,
      input_type_ids,
      config=bert_config,
      name='bert_model',
      float_type=float_type)

init_checkpoint="BERT_BASE_DIR/bert_model.ckpt"
logging.info(
            'Checkpoint file %s found and restoring from '
            'initial checkpoint for core model.', init_checkpoint)
checkpoint = tf.train.Checkpoint(model=sub_model)
checkpoint.restore(init_checkpoint).assert_consumed()
logging.info('Loading from checkpoint file completed')