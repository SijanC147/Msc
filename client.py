import argparse
import requests
import json
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from os import getcwd, listdir
from os.path import join, isfile
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.utils._nlp import get_sentence_contexts

parser = argparse.ArgumentParser(
    description="Run Target Sentiment Analysis on a sentence"
)

parser.add_argument(
    "--model",
    "-m",
    nargs=1,
    type=str,
    # required=True,
    help="Which model to use",
    default="lstm_changin_regression_outputs",
)

parser.add_argument(
    "--sentence",
    "-s",
    nargs=1,
    type=str,
    # required=True,
    help="Sentence to analyze, must contain the target",
    default="Hello my name is deep sean.",
)
parser.add_argument(
    "--target",
    "-t",
    nargs=1,
    type=str,
    # required=True,
    help="Target to focus on, must be in the sentence",
    default="sean",
)

args = parser.parse_args()

model = args.model[0]
sentence = args.sentence[0]
target = args.target[0]

export_directory = join(getcwd(), "export", model)
last_version = max(
    [int(v) for v in listdir(export_directory) if not isfile(v)]
)
embedding = Embedding(
    path=join(export_directory, str(last_version), "assets.extra", "embedding")
)

sentence = sentence.strip()
target = target.strip()

left_context, right_context = get_sentence_contexts(
    sentence=sentence, target=target
)

left_mapping = [embedding.get_index_ids(left_context)]
target_mapping = [embedding.get_index_ids(target)]
right_mapping = [embedding.get_index_ids(right_context)]

sentence_length = len(left_mapping + target_mapping + right_mapping)

sentence = sentence.encode()
left_context = left_context.encode()
right_context = right_context.encode()
target = target.encode()

context_features = tf.train.Features(
    feature={
        "sentence": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[sentence])
        ),
        "sentence_length": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[sentence_length])
        ),
        "left_literal": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[left_context])
        ),
        "right_literal": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[right_context])
        ),
        "target_literal": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[target])
        ),
    }
)
features_lists = tf.train.FeatureLists(
    feature_list={
        "left_mapping": tf.train.FeatureList(
            feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=mapping))
                for mapping in left_mapping
            ]
        ),
        "target_mapping": tf.train.FeatureList(
            feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=mapping))
                for mapping in target_mapping
            ]
        ),
        "right_mapping": tf.train.FeatureList(
            feature=[
                tf.train.Feature(int64_list=tf.train.Int64List(value=mapping))
                for mapping in right_mapping
            ]
        ),
    }
)

tf_example = tf.train.SequenceExample(
    feature_lists=features_lists, context=context_features
)
# print(tf_example)

serialized_tf_example = tf_example.SerializeToString()
# print(serialized_tf_example)

# URL = "http://localhost:8501/v1/models/lcro:classify"

# body = {"examples": [serialized_tf_example]}

# data = json.dumps(body)

# r = requests.post(url=URL, data=data)

# print(r.text)
