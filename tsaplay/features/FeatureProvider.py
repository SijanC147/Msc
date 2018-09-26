import tensorflow as tf
from os.path import join
from os import getcwd, makedirs
from tensorflow.train import BytesList, Feature, Features, Example, Int64List
from tensorflow.python_io import TFRecordWriter

import _constants as FEATURES


class FeatureProvider:
    def __init__(self, dataset, embedding):
        self.__dataset = dataset
        self.__embedding = embedding
        makedirs(self.gen_dir, exist_ok=True)

    @property
    def gen_dir(self):
        return join(
            FEATURES.DATA_PATH, self.__embedding.name, self.__dataset.name
        )

    def export_train_features(self):
        train_dict = self.__dataset.train_dict

        sentence_list = BytesList(
            value=[s.encode() for s in train_dict["sentences"]]
        )
        target_list = BytesList(
            value=[t.encode() for t in train_dict["targets"]]
        )
        label_list = Int64List(value=[int(l) for l in train_dict["labels"]])
        sentences = Feature(bytes_list=sentence_list)
        targets = Feature(bytes_list=target_list)
        labels = Feature(int64_list=label_list)

        training_feature = {
            "sentences": sentences,
            "targets": targets,
            "labels": labels,
        }

        training_dataset = Features(feature=training_feature)
        training_example = Example(features=training_dataset)

        tf_record_file = join(self.gen_dir, "train.tfrecord")

        with TFRecordWriter(tf_record_file) as tf_writer:
            tf_writer.write(training_example.SerializeToString())
