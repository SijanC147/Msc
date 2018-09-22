import time
import tensorflow as tf

from argparse import ArgumentParser

from grpc.beta import implementations

from tensorflow_serving.apis.input_pb2 import Input, ExampleList
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.classification_pb2 import ClassificationRequest
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto  # pylint: disable=E0611

from os import getcwd, listdir
from os.path import join, isfile

from tensorflow.train import (
    Example,
    SequenceExample,
    FeatureLists,
    FeatureList,
    Features,
    Feature,
    BytesList,
    Int64List,
)

from tsaplay.embeddings.Embedding import Embedding
from tsaplay.utils._nlp import get_sentence_target_features


def parse_args():

    parser = ArgumentParser(
        description="Run targeted sentiment analysis on a sentence."
    )

    parser.add_argument(
        "--model",
        "-m",
        dest="model",
        type=str,
        required=True,
        help="Which model to use",
    )

    parser.add_argument(
        "--phrase",
        "-p",
        dest="phrase",
        type=str,
        required=True,
        help="Phrase to analyze, must contain the target",
    )
    parser.add_argument(
        "--target",
        "-t",
        dest="target",
        type=str,
        required=True,
        help="Target to focus on, must be in the sentence",
    )

    args = parser.parse_args()

    return args.phrase, args.target, args.model


def get_export_embedding_path(model):
    export_dir = join(getcwd(), "export", model)
    versions = [int(v) for v in listdir(export_dir) if not isfile(v)]
    last_version = str(max(versions))
    assets_extra_path = join(export_dir, last_version, "assets.extra")
    embedding_path = join(assets_extra_path, "embedding")

    return embedding_path


def main():

    phrase, target, model = parse_args()

    export_embedding = Embedding(path=get_export_embedding_path(model))

    feat_dict = get_sentence_target_features(
        embedding=export_embedding, sentence=phrase, target=target
    )

    sentence = feat_dict["sentence"].encode()
    sen_length = feat_dict["sentence_len"]
    left_lit = feat_dict["left_lit"].encode()
    target = feat_dict["target_lit"].encode()
    right_lit = feat_dict["right_lit"].encode()
    left_map = feat_dict["left_map"]
    target_map = feat_dict["target_map"]
    right_map = feat_dict["right_map"]

    context = Features(
        feature={
            "sentence": Feature(bytes_list=BytesList(value=[sentence])),
            "sen_length": Feature(int64_list=Int64List(value=[sen_length])),
            "left_lit": Feature(bytes_list=BytesList(value=[left_lit])),
            "right_lit": Feature(bytes_list=BytesList(value=[right_lit])),
            "target_lit": Feature(bytes_list=BytesList(value=[target])),
            "left_map": Feature(int64_list=Int64List(value=left_map)),
            "target_map": Feature(int64_list=Int64List(value=target_map)),
            "right_map": Feature(int64_list=Int64List(value=right_map)),
        }
    )

    tf_example = Example(features=context)
    serialized = tf_example.SerializeToString()

    channel = implementations.insecure_channel(host="127.0.0.1", port=8500)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # CLASSIFICATION
    classification_req = ClassificationRequest()
    inputs = Input(example_list=ExampleList(examples=[tf_example]))
    classification_req.input.CopyFrom(inputs)  # pylint: disable=E1101
    classification_req.model_spec.name = "lcro"  # pylint: disable=E1101
    classification = stub.Classify(classification_req, 60.0)
    print(classification)

    # PREDICTION
    prediction_req = PredictRequest()
    tensor_proto = make_tensor_proto(serialized, dtype=tf.string, shape=[1])
    prediction_req.inputs["instances"].CopyFrom(  # pylint: disable=E1101
        tensor_proto
    )
    prediction_req.model_spec.name = "lcro"  # pylint: disable=E1101
    # prediction = stub.Predict(prediciton_req, 60.0)
    # print(prediction)


if __name__ == "__main__":
    main()
