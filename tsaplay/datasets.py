from os import makedirs
from os.path import join, exists
from tsaplay.utils.data import (
    class_dist_info,
    class_dist_stats,
    generate_corpus,
    resample_data_dict,
)
from tsaplay.utils.io import (
    unpickle_file,
    pickle_file,
    dump_json,
    list_folders,
)
from tsaplay.constants import DATASET_DATA_PATH


class Dataset:
    def __init__(self, name, redist=None):
        self._name = None
        self._redist = redist
        self._uid = None
        self._gen_dir = None
        self._train_dict = None
        self._test_dict = None
        self._train_corpus = None
        self._test_corpus = None
        self._train_srcdir = None
        self._test_srcdir = None
        self._train_dist = None
        self._test_dist = None
        self._train_dist_info = None
        self._test_dist_info = None
        self._class_labels = None

        self._init_gen_dir(name)
        for mode in ["train", "test"]:
            self._init_srcdirs(mode, redist)
            self._init_data_dict(mode, redist)
            self._init_dist_info(mode)
            self._init_corpus(mode)

        self._uid = "{name}-{train_dist}-{test_dist}".format(
            name=self._name,
            train_dist=self._train_dist,
            test_dist=self._test_dist,
        )

    @property
    def name(self):
        return self._name

    @property
    def redist(self):
        return self._redist

    @property
    def uid(self):
        return self._uid

    @property
    def gen_dir(self):
        return self._gen_dir

    @property
    def train_srcdir(self):
        return self._train_srcdir

    @property
    def test_srcdir(self):
        return self._test_srcdir

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
    def train_dist(self):
        return self._train_dist_info

    @property
    def test_dist(self):
        return self._test_dist_info

    @property
    def class_labels(self):
        return self._class_labels

    def _init_gen_dir(self, name):
        data_root = DATASET_DATA_PATH
        gen_dir = join(data_root, name)
        if not exists(gen_dir):
            available_datasets = list_folders(data_root)
            raise ValueError(
                """Expected  Dataset name to be one of {0},\
                got {1}. Import new datasets using\
                tsaplay.scripts.import_dataset""".format(
                    available_datasets, name
                )
            )
        self._gen_dir = gen_dir
        self._name = name

    def _init_srcdirs(self, mode, redist):
        redist = redist.get(mode) if isinstance(redist, dict) else redist
        srcdir_attr = "_{mode}_srcdir".format(mode=mode)
        srcdir = (
            join(
                self.gen_dir,
                "_redists",
                "_".join(map(str, map(int, redist))).replace(".", ""),
            )
            if redist
            else self.gen_dir
        )
        makedirs(srcdir, exist_ok=True)
        setattr(self, srcdir_attr, srcdir)

    def _init_data_dict(self, mode, redist):
        data_dict_file = "_{mode}_dict.pkl".format(mode=mode)
        data_dict_attr = "_{mode}_dict".format(mode=mode)
        redist = redist.get(mode) if isinstance(redist, dict) else redist
        srcdir_attr = "_{mode}_srcdir".format(mode=mode)
        data_dict_dir = getattr(self, srcdir_attr)
        data_dict_path = join(data_dict_dir, data_dict_file)
        if exists(data_dict_path):
            data_dict = unpickle_file(data_dict_path)
        elif redist:
            source_data_dict_path = join(self.gen_dir, data_dict_file)
            source_data_dict = unpickle_file(source_data_dict_path)
            data_dict = resample_data_dict(source_data_dict, redist)
            pickle_file(path=data_dict_path, data=data_dict)
        class_labels = self._class_labels or []
        class_labels = set(class_labels + data_dict["labels"])
        self._class_labels = list(class_labels)
        setattr(self, data_dict_attr, data_dict)

    def _init_corpus(self, mode):
        corpus_pkl_file = "_{mode}_corpus.pkl".format(mode=mode)
        srcdir_attr = "_{mode}_srcdir".format(mode=mode)
        corpus_pkl_dir = getattr(self, srcdir_attr)
        corpus_pkl_path = join(corpus_pkl_dir, corpus_pkl_file)
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

    def _init_dist_info(self, mode):
        dist_info_file = "_{mode}_dist.json".format(mode=mode)
        srcdir_attr = "_{mode}_srcdir".format(mode=mode)
        dist_info_dir = getattr(self, srcdir_attr)
        dist_info_path = join(dist_info_dir, dist_info_file)
        data_dict_attr = "_{mode}_dict".format(mode=mode)
        data_dict = getattr(self, data_dict_attr)
        _, _, dist_info = class_dist_info(data_dict["labels"])
        dist_key = "_".join(map(str, dist_info))
        dist_key_attr = "_{mode}_dist".format(mode=mode)
        setattr(self, dist_key_attr, dist_key)
        data = class_dist_stats(
            **{"classes": set(data_dict["labels"]), mode: data_dict}
        )
        dist_info_attr = "_{mode}_dist_info".format(mode=mode)
        setattr(self, dist_info_attr, data)
        if not exists(dist_info_path):
            dump_json(path=dist_info_path, data=data)
