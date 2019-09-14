import json
import re
import string
import sys
from collections import Counter

import numpy as np
import pandas as pd


# class Evaluate_helper(object):
#


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class Evaluate_helper(object):
    def __init__(self):
        self.read_data()

    def read_data(self):
        self.devContexOrigin = pd.read_pickle('./input/' + 'dev' + '_id2orign_context.pkl')
        self.devToken2Char = pd.read_pickle('./input/' + 'dev' + '_id2token2char.pkl')
        self.devContext = pd.read_pickle('./input/' + 'dev' + '_id2context.pkl')

        self.trContexOrigin = pd.read_pickle('./input/' + 'train' + '_id2orign_context.pkl')
        self.trToken2Char = pd.read_pickle('./input/' + 'train' + '_id2token2char.pkl')
        self.trContext = pd.read_pickle('./input/' + 'train' + '_id2context.pkl')

    def convert_predictions_to_answer(self, predictions, mode='train'):
        # print(predictions[0].shape, predictions[1].shape)
        # make class prediction
        if mode == 'train':
            ansBegin = np.zeros((len(predictions[0]),), dtype=np.int32)
            ansEnd = np.zeros((len(predictions[0]),), dtype=np.int32)
            for i in range(len(predictions[0])):
                ansBegin[i] = predictions[0][i, :].argmax()
                ansEnd[i] = predictions[1][i, :].argmax()
                # print(ansBegin.min(), ansBegin.max(), ansEnd.min(), ansEnd.max())
        else:
            ansBegin = []  # np.zeros((len(question_ids),), dtype=np.int32)
            ansEnd = []  # np.zeros((len(question_ids),), dtype=np.int32)
            for st in range(len(predictions[0])):
                for i in range(len(predictions[0][st])):
                    begin = predictions[0][st][i, :].argmax()
                    end = predictions[1][st][i, :].argmax()
                    if begin <= end:
                        ansBegin.append(begin)
                        ansEnd.append(end)
                    else:
                        mod_begin, mod_end = self.mod_convert_pred_to_answer(predictions[0][st][i, :],
                                                                             predictions[1][st][i, :])
                        ansBegin.append(mod_begin)
                        ansEnd.append(mod_end)

        return ansBegin, ansEnd

    def mod_convert_pred_to_answer(self, begin, end, return_origin=False):
        # modified version of function "convert_predictions_to_answer"
        # when meeting ansBegin
        if return_origin:
            ori_begin = begin.argmax()
            ori_end = end.argmax()

        begin = np.expand_dims(begin, axis=1)
        end = np.expand_dims(end, axis=0)
        conf = begin.dot(end)

        mask = np.zeros(conf.shape)
        i, j = np.indices(mask.shape)
        mask[i <= j] = 1

        conf *= mask
        number = conf.argmax()
        mod_start = number // conf.shape[0]
        mod_end = number % conf.shape[1]

        if return_origin:
            return mod_start, mod_end, ori_begin, ori_end
        else:
            return mod_start, mod_end

    def get_answer_span(self, ansBegin, ansEnd, question_ids, mode='train'):
        if mode == 'train':
            devContext = self.trContext
            devContexOrigin = self.trContexOrigin
            devToken2Char = self.trToken2Char
        else:
            devContext = self.devContext
            devContexOrigin = self.devContexOrigin
            devToken2Char = self.devToken2Char
        answers = {}
        for i, id in enumerate(question_ids):
            # print i
            if ansBegin[i] >= len(devContext[id]):
                answers[question_ids[i]] = ""
            elif ansEnd[i] >= len(devContext[id]):
                answers[question_ids[i]] = devContexOrigin[id][devToken2Char[id][ansBegin[i]]:]
            else:
                answers[question_ids[i]] = devContexOrigin[id][
                                           devToken2Char[id][ansBegin[i]]:devToken2Char[id][ansEnd[i]] + len(
                                               devContext[id][ansEnd[i]])]  # get span
        return answers

    def dump_answer(self, predictions, question_ids, mode='train'):
        ansBegin, ansEnd = self.convert_predictions_to_answer(predictions, mode)
        answers = self.get_answer_span(ansBegin, ansEnd, question_ids, mode)
        return answers

    def evaluate_dev(self, answers):
        # Since some question ask multi times
        expected_version = '1.1'
        dataset_file = './data/dev-v1.1.json'
        with open(dataset_file) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        f1 = exact_match = total = 0
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    total += 1
                    if qa['id'] not in answers:
                        message = 'Unanswered question ' + qa['id'] + \
                                  ' will receive score 0.'
                        print(message, file=sys.stderr)
                        continue
                    ground_truths = list(map(lambda x: x['text'], qa['answers']))
                    prediction = answers[qa['id']]
                    exact_match += metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths)
                    f1 += metric_max_over_ground_truths(
                        f1_score, prediction, ground_truths)

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total
        return exact_match, f1

    def evaluate_train(self, answers, ground_truths):
        f1 = exact_match = total = 0
        for qid, pred_answers in answers.items():
            total += 1
            groud_truth = ground_truths[qid]
            exact_match += metric_max_over_ground_truths(
                exact_match_score, pred_answers, [groud_truth])
            f1 += metric_max_over_ground_truths(
                f1_score, pred_answers, [groud_truth])
        exact_match = 100.0 * exact_match / total
        f1 = 100 * f1 / total
        return exact_match, f1


if __name__ == '__main__':
    expected_version = '1.1'
    dataset_file = './data/dev-v1.1.json'
    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

    prediction_file = './output/dynamitic_attention_enhanced_model__dev-prediction.json'
    with open(prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    print(json.dumps(evaluate(dataset, predictions)))
