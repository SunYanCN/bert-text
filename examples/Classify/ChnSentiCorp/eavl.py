import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from bert_text.layers import BertLayer
from bert_text.dataset import ChnSentiCorp
from bert.tokenization import FullTokenizer
from bert_text.prepare.classification import convert_examples_to_features

K.set_learning_phase(0)

model_path = "/home/CAIL/bert_text/examples/Classify/saved_models/mdoel.h5"
model = load_model(model_path,custom_objects=BertLayer().get_custom_objects())

dataset = ChnSentiCorp()


max_seq_length = 256

vocab_file = "/home/CAIL/bert_text/examples/SequenceLabel/vocab.txt"
tokenizer = FullTokenizer(vocab_file=vocab_file)
test_input_ids, test_input_masks, test_segment_ids, test_labels = convert_examples_to_features(tokenizer,
                                                                                                dataset.get_test_examples(),
                                                                                                max_seq_length=max_seq_length)
test_inputs = [test_input_ids, test_input_masks, test_segment_ids]
score = model.evaluate(test_inputs, test_labels, batch_size = 128)

print(score)

print("Test loss:", score[0]) #loss
print('Test accuracy:', score[1]) #accuracy
