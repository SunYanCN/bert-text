import os

import tensorflow as tf
import ujson as json
from bert import tokenization
from tqdm import tqdm

from dataset.dataset import SquadExample
from layers.utils import SimpleTokenizer
from utils.logger import logger

_DATA_URL = "https://github.com/SunYanCN/bert-text/raw/master/data/cmrc2018.tar.gz"


class CMRC:
    """
    A set of manually annotated Chinese word-segmentation data and
    specifications for training and testing a Chinese word-segmentation system
    for research purposes.  For more information please refer to
    https://www.microsoft.com/en-us/download/details.aspx?id=52531
    """

    def __init__(self, save_path="."):
        self.dataset_dir = os.path.join(save_path, "datasets/cmrc2018")
        if not os.path.exists(self.dataset_dir):
            file_path = tf.keras.utils.get_file(
                fname="cmrc2018.tar.gz",
                origin=_DATA_URL,
                extract=True,
                cache_dir=save_path,
            )
        else:
            logger.info("Dataset {} already cached.".format(self.dataset_dir))

        self._load_train_examples()
        self._load_dev_examples()
        self._load_test_examples()

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, "cmrc2018_train.json")
        self.train_examples = self._read_json(self.train_file, is_training=True, do_lower_case=False)
        self.train_num = len(self.train_examples)

    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.dataset_dir, "cmrc2018_dev.json")
        self.dev_examples = self._read_json(self.dev_file, is_training=True, do_lower_case=False)
        self.dev_num = len(self.dev_examples)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "cmrc2018_trial.json")
        self.test_examples = self._read_json(self.test_file, is_training=False, do_lower_case=False)
        self.test_num = len(self.test_examples)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return ["0", "1"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_json(self, input_file: str,
                   is_training: bool,
                   do_lower_case: bool):
        with tf.gfile.Open(input_file, "r") as reader:
            input_data = json.load(reader)["data"]

        examples = []

        for entry in tqdm(input_data, desc=input_file):
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                raw_doc_tokens = SimpleTokenizer(paragraph_text)
                doc_tokens = []
                char_to_word_offset = []

                k = 0
                temp_word = ""
                for c in paragraph_text:
                    if tokenization._is_whitespace(c):
                        char_to_word_offset.append(k - 1)
                        continue
                    else:
                        temp_word += c
                        char_to_word_offset.append(k)
                    if do_lower_case:
                        temp_word = temp_word.lower()
                    if temp_word == raw_doc_tokens[k]:
                        doc_tokens.append(temp_word)
                        temp_word = ""
                        k += 1

                assert k == len(raw_doc_tokens)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None

                    if is_training:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]

                        if orig_answer_text not in paragraph_text:
                            tf.logging.warning("Could not find answer")
                        else:
                            answer_offset = paragraph_text.index(orig_answer_text)
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]

                            # 跳过不符合要求的实例
                            actual_text = "".join(
                                doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = "".join(
                                tokenization.whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text,
                                                   cleaned_answer_text)
                                continue

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position)
                    examples.append(example)

        return examples

    def print_info(self, print_num=1):
        print("train examles:")
        for e in self.get_train_examples()[:print_num]:
            print("{}\t{}\t{}\t{}\t{}\t{}".format(e.qas_id, e.question_text, e.doc_tokens, e.orig_answer_text,
                                                  e.start_position, e.end_position))
        print("\n")

        print("dev examles:")
        for e in self.get_dev_examples()[:print_num]:
            print("{}\t{}\t{}\t{}\t{}\t{}".format(e.qas_id, e.question_text, e.doc_tokens, e.orig_answer_text,
                                                  e.start_position, e.end_position))
        print("\n")

        print("test examles:")
        for e in self.get_test_examples()[:print_num]:
            print("{}\t{}\t{}\t{}\t{}\t{}".format(e.qas_id, e.question_text, e.doc_tokens, e.orig_answer_text,
                                                  e.start_position, e.end_position))
        print("\n")

        print("Train number:{}, Dev number:{}, Test number:{}".format(self.train_num, self.dev_num, self.test_num))


if __name__ == "__main__":
    dataset = CMRC()
    # dataset.print_info(print_num=2)
