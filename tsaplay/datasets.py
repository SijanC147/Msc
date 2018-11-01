from os import makedirs
from os.path import join, exists, basename, normpath
from tsaplay.utils.data import (
    resample_data_dict,
    class_dist_info,
    class_dist_stats,
    generate_corpus,
)
from tsaplay.utils.io import search_dir, unpickle_file, pickle_file, dump_json
from tsaplay.utils.decorators import timeit
from tsaplay.constants import DATASET_DATA_PATH


class Dataset:
    def __init__(self, name, redist=None):
        self._name = None
        self._uid = None
        self._gen_dir = None
        self._train_dict = None
        self._test_dict = None
        self._train_corpus = None
        self._test_corpus = None
        self._train_dist = None
        self._test_dist = None
        self._class_labels = None

        self._init_gen_dir(name)
        for mode in ["train", "test"]:
            self._init_data_dict(mode, self.gen_dir)
            self._init_dist_info(mode, self.gen_dir)
            self._init_corpus(mode, self.gen_dir)

        # if redist:
        #     self._redistribute_data(redist)

        self._uid = "{name}-{train_dist}-{test_dist}".format(
            name=self._name,
            train_dist=self._train_dist,
            test_dist=self._test_dist,
        )

    @property
    def name(self):
        return self._name

    @property
    def uid(self):
        return self._uid

    @property
    def gen_dir(self):
        return self._gen_dir

    @property
    def train_dict(self):
        return self._train_dict

    @property
    def test_dict(self):
        return self._test_dict

    @property
    def train_corpus(self):
        return self._train_corpus

    @property
    def test_corpus(self):
        return self._test_corpus

    @property
    def class_labels(self):
        return self._class_labels

    def _init_gen_dir(self, name):
        data_root = DATASET_DATA_PATH
        gen_dir = join(data_root, name)
        if not exists(gen_dir):
            installed_datasets = [
                basename(normpath(path))
                for path in search_dir(data_root, kind="folders")
            ]
            raise ValueError(
                """Expected name to be one of {0}, got {1}.
                Import new datasets using
                tsaplay.scripts.import_dataset""".format(
                    installed_datasets, name
                )
            )
        else:
            self._gen_dir = gen_dir
            self._name = name

    def _init_data_dict(self, mode, path):
        data_dict_file = "_{mode}_dict.pkl".format(mode=mode)
        data_dict_path = join(path, data_dict_file)
        if not exists(data_dict_path):
            raise ValueError
        data_dict_attr = "_{mode}_dict".format(mode=mode)
        data_dict = unpickle_file(data_dict_path)
        class_labels = self._class_labels or []
        class_labels = set(class_labels + data_dict["labels"])
        self._class_labels = list(class_labels)
        setattr(self, data_dict_attr, data_dict)

    def _init_corpus(self, mode, path):
        corpus_pkl_file = "_{mode}_corpus.pkl".format(mode=mode)
        corpus_pkl_path = join(path, corpus_pkl_file)
        if exists(corpus_pkl_path):
            corpus = unpickle_file(corpus_pkl_path)
        else:
            dict_attr = "_{mode}_dict".format(mode=mode)
            data_dict = getattr(self, dict_attr)
            docs = data_dict["sentences"]
            corpus = generate_corpus(docs, mode)
            pickle_file(data=corpus, path=corpus_pkl_path)
        corpus_attr = "_{mode}_corpus".format(mode=mode)
        setattr(self, corpus_attr, corpus)

    def _init_dist_info(self, mode, path):
        dist_info_file = "_{mode}_dist.json".format(mode=mode)
        dist_info_path = join(path, dist_info_file)
        data_dict_attr = "_{mode}_dict".format(mode=mode)
        data_dict = getattr(self, data_dict_attr)
        _, _, dist_info = class_dist_info(data_dict["labels"])
        dist_key = "_".join(map(str, dist_info))
        dist_key_attr = "_{mode}_dist".format(mode=mode)
        setattr(self, dist_key_attr, dist_key)
        if not exists(dist_info_path):
            data = {"classes": set(data_dict["labels"]), mode: data_dict}
            dump_json(path=dist_info_path, data=class_dist_stats(**data))

    @timeit("Redistributing dataset", "Dataset redistributed")
    def _redistribute_data(self, distribution):
        dists_dir = join(self.gen_dir, "_dists")
        makedirs(dists_dir, exist_ok=True)
        if isinstance(distribution, list):
            dist_list = distribution
            distribution = {"train": dist_list, "test": dist_list}
        elif not isinstance(distribution, dict):
            raise ValueError
        for key, dist_values in distribution.items():
            dist_folder = "_".join([str(int(v * 100)) for v in dist_values])
            dist_path = join(dists_dir, dist_folder)
            makedirs(dist_path, exist_ok=True)
            dist_dict_path = join(dist_path, "_" + key + "_dict.pkl")
            dist_corpus_path = join(dist_path, "_" + key + "_corpus.pkl")
            if exists(dist_dict_path):
                resampled_dict = unpickle_file(path=dist_dict_path)
                if key == "train":
                    self._train_dict = resampled_dict
                else:
                    self._test_dict = resampled_dict
            else:
                data_dicts = {}
                for mode in ["train", "test"]:
                    dist_dict_path = join(dist_path, "_" + mode + "_dict.pkl")
                    if mode == "train":
                        orig_dict = self._train_dict
                    else:
                        orig_dict = self._test_dict

                    resampled_dict = resample_data_dict(orig_dict, dist_values)
                    resampled_docs = set(resampled_dict["sentences"])
                    resampled_corpus = generate_corpus(resampled_docs)
                    pickle_file(path=dist_dict_path, data=resampled_dict)
                    pickle_file(path=dist_corpus_path, data=resampled_corpus)
                    data_dicts[mode] = resampled_dict

                dump_json(
                    path=join(dist_path, "_stats.json"),
                    data=class_dist_stats(
                        classes=self.class_labels, **data_dicts
                    ),
                )

                if key == "train":
                    self._train_dict = data_dicts[key]
                else:
                    self._test_dict = data_dicts[key]
