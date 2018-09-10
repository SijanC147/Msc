# Tensorflow NLP Sentiment Analysis Playground

A codebase to bring together different embeddings, datasets and models and efficiently carry out experiments with them. 

## Getting Started 

### Pre-requisites 

The project uses

*   Python 3.6
*   Pipenv 

### Installing 

Setup should be straight forward enough using `pipenv`, from the directory of the project run the following command 

````Bash
pipenv install
````

### Running Experiments

Setting up experiments takes place in `main.py`. 

Each `Experiment` object takes the following 
*   An `Embedding`
*   A `Dataset`
*   A `Model`
*   A `RunConfig` (optional)

Experiments can then be run on an experiment instance using `experiment.run(job, steps)`

The job options avaiable are
*   `'train'`, steps **must** be provided.
*   `'eval'`, model **must** have been previously trained. 
*   `'train+eval'`, steps **must** be provided, **This is the most easy and straight forward approach**


## Experiment Process

### Embedding

All embedding files need to have a path as follows: `embeddings\data\<name>\<version>.txt`

The **path** as shown above is passed on to the `Embedding` constructor

The **name** and **version** parts of the path are assigned to the Embedding object internally as identifiers. 

### Dataset

Datasets are initialized with a path as follows: `datasets\data\<name>\`

The **path** and a **parser** is passed on to the `Dataset` constructor

Internally the system looks for the first files in that directory with the words `train` and `test` in there name.

The system then parses these files with the **specified parser** to generate all the required features and labels. 

### Model

All models should be defined under `models\<can_be_whatever>\<model>.py`

All models must inherit from the base `Model` class

All models must contain implementations for the following functions: 

*   `_params`
*   `_feature_columns`
*   `_train_input_fn`
*   `_eval_input_fn`
*   `_model_fn`

All of these functions must return the respective parts of the tensorflow model. 

Everything else is taken care of internally by the system.

### Experiment Results

After running an experiment, results are written to `experiments\data\<can_be_whatever>\<model>\`

The directory contains some extra statistics files, and markdown of functions used for future reference. 

The **Tensorboard logdir** is named `tb_summary\`

To launch tensorboard after an experiment run the following, 

````Bash
tensorboard --logdir experiments\data\<can_be_whatever>\<model>\tb_summary
````

Alternatively, the `experiment.run()` method takes an optional `start_tb` boolean parameter. **MAC ONLY**

If set to true, the process will launch the tensorboard page and start the tensorboard process automatically. 

The tensorboard page will fail to load from the get go until the process starts, it should reload automatically within a few seconds. 

## Performance 

Extracting features and parsing the dataset files takes a while.

To help with this, the `Dataset` class internally generates a myriad of files during its initial execution to use them in future runs. 

All of the generated files are stored under `datasets\data\<name>\generated`

The files saved include: 

*   `corpus.csv` 
    *   A comma seperated file of all unique tokens appearing in the dataset, and their word count
*   `train_dict.pkl` and `test_dict.pkl`
    *   Pickle binary files of the parsed datasets in dictionary format
    *   These would still need to be mapped to indices of a specific embedding
*   `<embedding_name>\partial_<embedding_version>.txt`
    *   A filtered down version of an embedding containing only the words that are in the dataset corpus
    *   This will be loaded instead of the full embedding in future runs
*  `<embedding_name>\projection_meta.tsv`
    *   Can be loaded into the tensorboard projection tab as labels to when viewing an embedding.
*  `<embedding_name>\train.pkl` and `<embedding_name>\test.pkl`
    *   These are the actual embedding IDs for the dataset
    *   This process takes hours the first time, but subsequently features are loaded instantly through these files.
    *   Naturall, these files remain valid so long as the partial file remains the same, otherwise the IDs will not reflect the correct words in the embedding.

## Notes

I am working on generating the files mentioned above and making them available online, or in the repo itself, so you don't have to go through the parsing process the first time.
