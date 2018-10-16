import argparse
import imp
import inspect
from functools import wraps
from shutil import rmtree
from os import makedirs
from os.path import join, exists, normpath, basename
from tsaplay.utils.io import pickle_file, search_dir, cprnt
from tsaplay.utils.decorators import wrap_parsing_fn
from tsaplay.constants import DATASET_DATA_PATH
from tsaplay.datasets import Dataset

IMPORTER_MODULE_NAME = "tsaplay_dataset_importer"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str, help="Path to the dataset files")

    parser.add_argument(
        "--dataset-name", type=str, help="Name to register the dataset under"
    )
    parser.add_argument(
        "--parser-name",
        type=str,
        help="Name of the parsing python script for the dataset",
        default=None,
    )

    return parser.parse_args()


def get_dataset_dicts(train_file, test_file, parsing_fn):
    train_dict = parsing_fn(train_file)
    test_dict = parsing_fn(test_file)
    return train_dict, test_dict


def get_raw_file_paths(path):
    train_file = search_dir(path, "train", first=True, kind="files")
    test_file = search_dir(path, "test", first=True, kind="files")
    return train_file, test_file


def get_parsing_fn(path, parser_name=None):
    parser_name = parser_name or "parser"
    parser_path = join(path, parser_name + ".py")
    if not exists(parser_path):
        raise ValueError("Parser file {} does not exist.".format(parser_path))

    parser_module = imp.load_source(
        name=IMPORTER_MODULE_NAME, pathname=parser_path
    )

    parsing_fn_candidates = get_parsing_fn_candidates(parser_module)

    parsing_fns = [
        fn for fn in parsing_fn_candidates if is_valid_parsing_fn(fn[1])
    ]

    num_parsing_fns = len(parsing_fns)
    if num_parsing_fns is not 1 and parser_name is not None:
        parsing_fns = [fn for fn in parsing_fns if fn[0] == parser_name]
        num_parsing_fns = len(parsing_fns)
    if num_parsing_fns is not 1:
        raise ValueError(
            "Expected 1 valid parsing fn in {0} found {1}".format(
                parser_path, num_parsing_fns
            )
        )

    return wrap_parsing_fn(parsing_fns[0][1])


def get_parsing_fn_candidates(parser_module):
    return inspect.getmembers(parser_module, predicate=parsing_fn_predicate)


def parsing_fn_predicate(module_member):
    return inspect.isfunction(module_member) and not inspect.isbuiltin(
        module_member
    )


def is_valid_parsing_fn(parsing_fn_candidate):
    if parsing_fn_candidate.__module__ is not IMPORTER_MODULE_NAME:
        return False

    fn_signature = inspect.signature(parsing_fn_candidate)
    fn_parameters = fn_signature.parameters

    params = fn_parameters.values()

    valid_params = [
        param for param in params if param.default is inspect.Parameter.empty
    ]

    return len(valid_params) == 1


def main(args):
    dataset_name = args.dataset_name or basename(normpath(args.path))
    parsing_fn = get_parsing_fn(args.path, args.parser_name)
    ftrain, ftest = get_raw_file_paths(args.path)
    train_dict, test_dict = get_dataset_dicts(ftrain, ftest, parsing_fn)
    all_docs = set(train_dict["sentences"] + test_dict["sentences"])
    target_path = join(DATASET_DATA_PATH, dataset_name)
    if exists(target_path):
        rmtree(target_path)
    makedirs(target_path)
    Dataset.write_stats_json(target_path, train=train_dict, test=test_dict)
    Dataset.generate_corpus(all_docs, target_path)
    pickle_file(join(target_path, "_train.pkl"), train_dict)
    pickle_file(join(target_path, "_test.pkl"), test_dict)


if __name__ == "__main__":
    main(get_args())
