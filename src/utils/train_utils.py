import os
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def batch_generator(inputs, labels, shuffle=False, batch_size=10):
    unique_ids, input_ids, input_masks, segment_ids = inputs
    start_positions, end_positions = labels

    number_examples = unique_ids.shape[0]
    while 1:
        if shuffle:
            new_index = np.random.permutation(number_examples)
            unique_ids = unique_ids[new_index]
            input_ids = input_ids[new_index]
            input_masks = input_masks[new_index]
            segment_ids = segment_ids[new_index]
            start_positions = start_positions[new_index]
            end_positions = end_positions[new_index]

        batch_num = int(number_examples / batch_size)

        for batch in range(batch_num):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, number_examples)

            batch_inputs = [unique_ids[start:end], input_ids[start:end], input_masks[start:end], segment_ids[start:end]]
            batch_labels = [start_positions[start:end], end_positions[start:end]]

            yield batch_inputs, batch_labels


class BatchGenerator(object):
    def __init__(self, inputs, labels, shuffle=False):

        self._unique_ids, self._input_ids, self._input_masks, self._segment_ids = inputs
        self._start_positions, self._end_positions = labels

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._unique_ids.shape[0]
        self._shuffle = shuffle

        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._unique_ids = self._unique_ids[new_index]
            self._input_ids = self._input_ids[new_index]
            self._input_masks = self._input_masks[new_index]
            self._segment_ids = self._segment_ids[new_index]
            self._start_positions = self._start_positions[new_index]
            self._end_positions = self._end_positions[new_index]

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._unique_ids = self._unique_ids[new_index]
                self._input_ids = self._input_ids[new_index]
                self._input_masks = self._input_masks[new_index]
                self._segment_ids = self._segment_ids[new_index]
                self._start_positions = self._start_positions[new_index]
                self._end_positions = self._end_positions[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch

        batch_inputs = [self._unique_ids[start:end], self._input_ids[start:end], self._input_masks[start:end],
                        self._segment_ids[start:end]]
        batch_labels = [self._start_positions, self._end_positions]

        return batch_inputs, batch_labels


class Trainer_helper(object):
    def __init__(self, model, log_dir='./logs/', bst_model_path='./model/squad_rnet.hdf5', output_name='./',
                 evaluate_helper=None):
        """
       :param model:
       :param log_dir:  log path
       :param bst_model_path: model path
       :param output_name: prediction save path
       :param evaluate_helper:  evaluater helper for evaluate train on batch and dev data
       """
        self.model = model
        if os.path.exists(log_dir) == False:
            os.mkdir(log_dir)
        self.bst_model_path = bst_model_path
        self.output_name = output_name
        self.tf_train_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'train'))
        self.tf_test_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'dev'))
        self.set_train_generator()  # default
        self.set_valid_generator()  # default
        # self.evaluate_helper = Evaluate_helper()
        # self.train_groud_truth = pd.read_pickle('./input/train_id2answer.pkl')
        self.is_save_intemediate = False

    def set_train_generator(self, gen=None):
        if gen is not None:
            self.tr_gen = gen
        else:
            self.tr_gen = BatchGenerator

    def set_valid_generator(self, gen=None):
        if gen is not None:
            self.te_gen = gen
        else:
            self.te_gen = BatchGenerator

    def save_logs(self, em, f1, i, mode='train'):
        logs = {'Extract_Match': em, 'F1': f1}
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            if mode == 'train':
                self.tf_train_writer.add_summary(summary, i)
            else:
                self.tf_test_writer.add_summary(summary, i)

        if mode == 'train':
            self.tf_train_writer.flush()
        else:
            self.tf_test_writer.flush()

    def fit(self, train_x, train_y, valid_x, valid_y,
            n_epoch, batch_size=1, dev_batch_size=10, early_stop=1, verbose_train=200, is_save_intermediate=False,
            save_epoch=False, adjust_lr=False):

        best_em = 0
        best_f1 = 0
        early_stop_step = 0

        for i in tqdm(range(n_epoch), desc="Epoch Completed"):
            tr_gen = self.tr_gen(train_x, train_y, shuffle=False)
            steps_per_epoch = int(np.ceil(len(train_x[0]) / float(batch_size)))

            loss = 0
            answer_start_loss = 0
            answer_end_loss = 0
            answer_start_acc = 0
            answer_end_acc = 0
            cnt = 0
            cnt_tr = 0
            em_tr = 0
            f1_tr = 0

            for X_batch, Y_batch in tr_gen.next_batch(batch_size=batch_size):
                # X_ids = X_batch[0]

                val = self.model.train_on_batch(X_batch, Y_batch)
                element_cnt = X_batch[0].shape[0]
                loss += val[0] * element_cnt
                answer_start_loss += val[1] * element_cnt
                answer_end_loss += val[2] * element_cnt
                answer_start_acc += val[3] * element_cnt
                answer_end_acc += val[4] * element_cnt
                cnt += element_cnt
                print(
                    '----train loss:%f,answer start loss:%f,answer end loss:%f,answer start acc:%f,answer end acc:%f-------' % (
                        loss / cnt,
                        answer_start_loss / cnt,
                        answer_end_loss / cnt,
                        answer_start_acc / cnt,
                        answer_end_acc / cnt))

                # if step % verbose_train == 0:
                #     predictions = self.model.predict(X_batch)
                #     answer = self.evaluate_helper.dump_answer(
                #         predictions, X_ids, mode='train')
                #     t_em, t_f1 = self.evaluate_helper.evaluate_train(
                #         answer, self.train_groud_truth)
                #     em_tr += t_em * element_cnt
                #     f1_tr += t_f1 * element_cnt
                #     cnt_tr += element_cnt
                #     print('--train Extract Match score :%f--------F1 score:%f' %
                #           (em_tr / cnt_tr, f1_tr / cnt_tr))
                #     self.save_logs(em_tr / cnt_tr, f1_tr / cnt_tr, i *
                #                    steps_per_epoch + step, mode='train')  # total step

                sys.stdout.flush()

    #         em, f1 = self.evaluate_on_dev(
    #             valid_x, dev_batch_size, i)  # can control huge data
    #         print('--Dev Extract Match score :%f--------F1 score:%f' % (em, f1))
    #
    #         self.save_logs(em, f1, i, mode='dev')
    #         if em > best_em or f1 > best_f1:
    #             self.model.save_weights(self.bst_model_path)
    #             print('save model to ', self.bst_model_path)
    #             if save_epoch:
    #                 self.model.save_weights(
    #                     self.bst_model_path.replace('.hdf5', '.ep%d.hdf5' % i))
    #                 print('save model to ', self.bst_model_path.replace(
    #                     '.hdf5', '.ep%d.hdf5' % i))
    #             if em > best_em:
    #                 best_em = em
    #             if f1 > best_f1:
    #                 best_f1 = f1
    #             early_stop_step = 0
    #         else:
    #             early_stop_step += 1
    #             if early_stop_step >= early_stop:
    #                 print('early stop @', i)
    #                 break
    #             if adjust_lr:
    #                 lr = float(K.get_value(self.model.optimizer.lr))
    #                 K.set_value(self.model.optimizer.lr, lr / 2)
    #
    # def get_predict(self, valid_x, batch_size):
    #     te_gen = self.te_gen(valid_x, batch_size=batch_size)
    #     steps = int(np.ceil(len(valid_x[0]) / float(batch_size)))
    #     st = []
    #     ed = []
    #     for X, step in tqdm(zip(te_gen, range(steps)), total=steps):
    #         # batch on predict acclerate betten on predict
    #         _st, _ed = self.model.predict_on_batch(X)
    #         st.append(_st)
    #         ed.append(_ed)
    #     return [st, ed]
    #
    # def evaluate_on_dev(self, valid_x, batch_size, epoch_idx):
    #     predictions = self.get_predict(valid_x, batch_size)
    #     if self.is_save_intemediate:
    #         pd.to_pickle(predictions, epoch_idx + '_' + self.output_name)
    #     dev_ids = pd.read_pickle('./input/dev_question_id.pkl')
    #     answers = self.evaluate_helper.dump_answer(predictions, dev_ids, mode='dev')
    #     return self.evaluate_helper.evaluate_dev(answers)
    #
    # def predict_probs(self, valid_x, batch_size):
    #     predictions = self.get_predict(valid_x, 10)
    #     return predictions
    #
    # def predict(self, valid_x, batch_size):
    #     predictions = self.get_predict(valid_x, 10)
    #     dev_ids = pd.read_pickle('./input/dev_question_id.pkl')
    #     answers = self.evaluate_helper.dump_answer(
    #         predictions, dev_ids, mode='dev')
    #     return answers
