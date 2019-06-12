import tensorflow as tf

class BertSquadLogitsLayer(tf.keras.layers.Layer):
  """Returns a layer that computes custom logits for BERT squad model."""

  def __init__(self, initializer=None, float_type=tf.float32, **kwargs):
    super(BertSquadLogitsLayer, self).__init__(**kwargs)
    self.initializer = initializer
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.final_dense = tf.keras.layers.Dense(
        units=2, kernel_initializer=self.initializer, name='final_dense')
    super(BertSquadLogitsLayer, self).build(unused_input_shapes)

  def call(self, inputs, **kwargs):
    """Implements call() for the layer."""
    sequence_output = inputs

    input_shape = sequence_output.shape.as_list()
    sequence_length = input_shape[1]
    num_hidden_units = input_shape[2]

    final_hidden_input = tf.keras.backend.reshape(sequence_output,
                                                  [-1, num_hidden_units])
    logits = self.final_dense(final_hidden_input)
    logits = tf.keras.backend.reshape(logits, [-1, sequence_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])
    unstacked_logits = tf.unstack(logits, axis=0)
    return unstacked_logits[0], unstacked_logits[1]

  def squad_loss_fn(self,start_positions,
                    end_positions,
                    start_logits,
                    end_logits,
                    loss_scale=1.0):
    """Returns sparse categorical crossentropy for start/end logits."""
    start_loss = tf.keras.backend.sparse_categorical_crossentropy(
      start_positions, start_logits, from_logits=True)
    end_loss = tf.keras.backend.sparse_categorical_crossentropy(
      end_positions, end_logits, from_logits=True)

    total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
    total_loss *= loss_scale
    return total_loss

  def loss(self, y_true, y_pred):
    pass
