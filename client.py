import tensorflow as tf

from argparse import ArgumentParser

from grpc import insecure_channel

from tensorflow_serving.apis.input_pb2 import Input, ExampleList
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.classification_pb2 import ClassificationRequest
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.contrib.util import make_tensor_proto  # pylint: disable=E0611

from os import getcwd, listdir
from os.path import join, isfile, splitext
from csv import DictReader

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

from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.utils._nlp import get_sentence_target_features
from tsaplay.utils._data import tf_encoded_tokenisation


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
        "--batch_file",
        "-b",
        dest="batch_file",
        type=str,
        required=False,
        help="Process batch of targets,phrases in file",
    )

    parser.add_argument(
        "--phrase",
        "-p",
        dest="phrase",
        type=str,
        required=False,
        help="Phrase to analyze, must contain the target",
    )
    parser.add_argument(
        "--target",
        "-t",
        dest="target",
        type=str,
        required=False,
        help="Target to focus on, must be in the sentence",
    )

    args = parser.parse_args()

    return args.phrase, args.target, args.model, args.batch_file


def make_example(feat_dict):
    sen_tok = tf_encoded_tokenisation([feat_dict["sentence"]], to_bytes=True)
    left_tok = tf_encoded_tokenisation([feat_dict["left_lit"]], to_bytes=True)
    right_tok = tf_encoded_tokenisation(
        [feat_dict["right_lit"]], to_bytes=True
    )
    target_tok = tf_encoded_tokenisation(
        [feat_dict["target_lit"]], to_bytes=True
    )
    sentence = feat_dict["sentence"].encode()
    sen_length = feat_dict["sentence_len"]
    left_lit = feat_dict["left_lit"].encode()
    target = feat_dict["target_lit"].encode()
    right_lit = feat_dict["right_lit"].encode()
    left_map = feat_dict["left_map"]
    target_map = feat_dict["target_map"]
    right_map = feat_dict["right_map"]
    left_len = len(left_map)
    right_len = len(right_map)
    target_len = len(target_map)
    sen_map = left_map + target_map + right_map
    ctxt_map = left_map + right_map
    lft_trg_map = left_map + target_map
    trg_rht_map = list(reversed(target_map + right_map))

    if left_len == 0:
        left_map += [0]
    if right_len == 0:
        right_map += [0]

    context = Features(
        feature={
            "sen_lit": Feature(bytes_list=BytesList(value=[sentence])),
            "left_lit": Feature(bytes_list=BytesList(value=[left_lit])),
            "right_lit": Feature(bytes_list=BytesList(value=[right_lit])),
            "target_lit": Feature(bytes_list=BytesList(value=[target])),
            "sen_tok": Feature(bytes_list=BytesList(value=sen_tok)),
            "left_tok": Feature(bytes_list=BytesList(value=left_tok)),
            "right_tok": Feature(bytes_list=BytesList(value=right_tok)),
            "target_tok": Feature(bytes_list=BytesList(value=target_tok)),
            "sen_len": Feature(int64_list=Int64List(value=[sen_length])),
            "left_len": Feature(int64_list=Int64List(value=[left_len])),
            "right_len": Feature(int64_list=Int64List(value=[right_len])),
            "target_len": Feature(int64_list=Int64List(value=[target_len])),
            "sen_map": Feature(int64_list=Int64List(value=sen_map)),
            "right_map": Feature(int64_list=Int64List(value=right_map)),
            "left_map": Feature(int64_list=Int64List(value=left_map)),
            "target_map": Feature(int64_list=Int64List(value=target_map)),
            "ctxt_map": Feature(int64_list=Int64List(value=ctxt_map)),
            "lft_trg_map": Feature(int64_list=Int64List(value=lft_trg_map)),
            "trg_rht_map": Feature(int64_list=Int64List(value=trg_rht_map)),
        }
    )

    return Example(features=context)


def main():

    phrase, target, model, batch_file = parse_args()

    export_embedding = Embedding(source=EMBEDDINGS.DEBUG)

    if batch_file is not None:
        tf_examples = []
        phrases = []
        targets = []
        batch_file_ext = splitext(batch_file)[1]
        if batch_file_ext == ".txt":
            with open(batch_file, "r") as f:
                for line in f:
                    target, _, phrase = line.partition(",")
                    phrases.append(phrase.strip())
                    targets.append(target.strip())
        elif batch_file_ext == ".csv":
            with open(batch_file, "r") as f:
                csvreader = DictReader(f, fieldnames=["target", "phrase"])
                for row in csvreader:
                    phrases.append(row["phrase"].strip())
                    targets.append(row["target"].strip())
        else:
            print("supported batch_file types: txt, csv")
            return

        for (target, phrase) in zip(targets, phrases):
            feat_dict = get_sentence_target_features(
                embedding=export_embedding, sentence=phrase, target=target
            )
            tf_example = make_example(feat_dict)
            tf_examples.append(tf_example.SerializeToString())
        tensor_proto = make_tensor_proto(
            tf_examples, dtype=tf.string, shape=[len(tf_examples)]
        )

    else:
        feat_dict = get_sentence_target_features(
            embedding=export_embedding, sentence=phrase, target=target
        )
        tf_example = make_example(feat_dict)
        serialized = tf_example.SerializeToString()
        tensor_proto = make_tensor_proto(
            serialized, dtype=tf.string, shape=[1]
        )

    channel = insecure_channel("127.0.0.1:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # CLASSIFICATION
    # classification_req = ClassificationRequest()
    # inputs = Input(example_list=ExampleList(examples=[tf_example]))
    # classification_req.input.CopyFrom(inputs)  # pylint: disable=E1101
    # classification_req.model_spec.name = "lg"  # pylint: disable=E1101
    # classification = stub.Classify(classification_req, 60.0)
    # print(classification)

    # PREDICTION
    prediction_req = PredictRequest()
    prediction_req.inputs["instances"].CopyFrom(  # pylint: disable=E1101
        tensor_proto
    )
    prediction_req.model_spec.signature_name = (  # pylint: disable=E1101
        "inspect"
    )
    prediction_req.model_spec.name = "lg"  # pylint: disable=E1101
    prediction = stub.Predict(prediction_req, 60.0)
    print(prediction)


if __name__ == "__main__":
    main()
