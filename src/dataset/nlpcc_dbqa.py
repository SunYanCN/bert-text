import codecs
import csv
import os

import pandas as pd
import tensorflow as tf

from dataset.dataset import InputExample
from utils.logger import logger

_DATA_URL = "https://paddlehub-dataset.bj.bcebos.com/nlpcc-dbqa.tar.gz"


class NLPCC_DBQA:
    """
    A set of manually annotated Chinese word-segmentation data and
    specifications for training and testing a Chinese word-segmentation system
    for research purposes.  For more information please refer to
    https://www.microsoft.com/en-us/download/details.aspx?id=52531
    """

    def __init__(self, save_path=".", load_df=False):
        self.dataset_dir = os.path.join(save_path, "datasets/nlpcc-dbqa")
        if not os.path.exists(self.dataset_dir):
            file_path = tf.keras.utils.get_file(
                fname="nlpcc-dbqa.tar.gz",
                origin=_DATA_URL,
                extract=True,
                cache_dir=save_path,
            )
        else:
            logger.info("Dataset {} already cached.".format(self.dataset_dir))

        if load_df:
            self._load_dataset_df()
        else:
            self._load_train_examples()
            self._load_test_examples()
            self._load_dev_examples()

    def _load_dataset_df(self):
        self.train_file = os.path.join(self.dataset_dir, "train.tsv")
        self.dev_file = os.path.join(self.dataset_dir, "dev.tsv")
        self.test_file = os.path.join(self.dataset_dir, "test.tsv")

        self.train_df = pd.read_csv(self.train_file, sep="\t")
        self.dev_df = pd.read_csv(self.dev_file, sep="\t")
        self.test_df = pd.read_csv(self.test_file, sep="\t")

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, "train.tsv")
        self.train_examples = self._read_tsv(self.train_file)
        self.train_num = len(self.train_examples)

    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.dataset_dir, "dev.tsv")
        self.dev_examples = self._read_tsv(self.dev_file)
        self.dev_num = len(self.dev_examples)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "test.tsv")
        self.test_examples = self._read_tsv(self.test_file)
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

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            examples = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                example = InputExample(
                    guid=seq_id, label=line[3], text_a=line[1], text_b=line[2])
                seq_id += 1
                examples.append(example)

            return examples

    def print_info(self, print_num=1):
        print("train examles:")
        for e in self.get_train_examples()[:print_num]:
            print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
        print("\n")

        print("dev examles:")
        for e in self.get_dev_examples()[:print_num]:
            print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
        print("\n")

        print("test examles:")
        for e in self.get_test_examples()[:print_num]:
            print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
        print("\n")

        print("Train number:{}, Dev number:{}, Test number:{}".format(self.train_num, self.dev_num, self.test_num))


if __name__ == "__main__":
    dataset = NLPCC_DBQA()
    dataset.print_info(print_num=2)
