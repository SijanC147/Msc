from argparse import ArgumentParser
from csv import DictReader
from grpc import insecure_channel

from tensorflow import string as tf_string
from tensorflow.contrib.util import make_tensor_proto  # pylint: disable=E0611
from tensorflow.train import Example, Features, Feature, BytesList
from tensorflow_serving.apis.input_pb2 import Input, ExampleList
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.classification_pb2 import ClassificationRequest
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tsaplay.features import FeatureProvider


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
        "--signature",
        "-sig",
        dest="signature",
        default="inspect",
        type=str,
        required=False,
        help="Specific TF Serving signature to query.",
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
        "--sentence",
        "-s",
        dest="sentence",
        type=str,
        required=False,
        help="Sentence to analyze, must contain the target",
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

    return (
        args.model,
        args.signature,
        args.batch_file,
        args.sentence,
        args.target,
    )


def byte_encode_array(array):
    return [a.encode() for a in array]


def main():

    model, signature, batch_file_path, sentence, target = parse_args()

    feat_dict = {"sentences": [], "targets": []}

    if batch_file_path is not None:
        with open(batch_file_path, "r") as batch_file:
            fieldnames = ["target", "sentence"]
            csvreader = DictReader(batch_file, fieldnames=fieldnames)
            for row in csvreader:
                feat_dict["targets"].append(row["target"].strip())
                feat_dict["sentences"].append(row["sentence"].strip())
    else:
        feat_dict["targets"].append(target)
        feat_dict["sentences"].append(sentence)

    l_ctxts, trgs, r_ctxts = FeatureProvider.partition_sentences(
        sentences=feat_dict["sentences"],
        targets=feat_dict["targets"],
        offsets=FeatureProvider.get_target_offset_array(feat_dict),
    )
    l_enc = [
        FeatureProvider.tf_encode_tokens(tokens)
        for tokens in FeatureProvider.tokenize_phrases(l_ctxts)
    ]
    trg_enc = [
        FeatureProvider.tf_encode_tokens(tokens)
        for tokens in FeatureProvider.tokenize_phrases(trgs)
    ]
    r_enc = [
        FeatureProvider.tf_encode_tokens(tokens)
        for tokens in FeatureProvider.tokenize_phrases(r_ctxts)
    ]

    tf_examples = []

    for left, target, right in zip(l_enc, trg_enc, r_enc):
        features = Features(
            feature={
                "left": Feature(bytes_list=BytesList(value=left)),
                "target": Feature(bytes_list=BytesList(value=target)),
                "right": Feature(bytes_list=BytesList(value=right)),
            }
        )
        tf_example = Example(features=features)
        tf_examples.append(tf_example.SerializeToString())

    tensor_proto = make_tensor_proto(
        tf_examples, dtype=tf_string, shape=[len(tf_examples)]
    )

    channel = insecure_channel("127.0.0.1:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # CLASSIFICATION
    classification_req = ClassificationRequest()
    inputs = Input(example_list=ExampleList(examples=[tf_example]))
    classification_req.input.CopyFrom(inputs)  # pylint: disable=E1101
    classification_req.model_spec.name = "lg"  # pylint: disable=E1101
    classification = stub.Classify(classification_req, 60.0)
    print(classification)

    # PREDICTION
    prediction_req = PredictRequest()
    prediction_req.inputs["instances"].CopyFrom(  # pylint: disable=E1101
        tensor_proto
    )
    prediction_req.model_spec.signature_name = (  # pylint: disable=E1101
        signature
    )
    prediction_req.model_spec.name = model  # pylint: disable=E1101
    prediction = stub.Predict(prediction_req, 60.0)
    print(prediction)


if __name__ == "__main__":
    main()
