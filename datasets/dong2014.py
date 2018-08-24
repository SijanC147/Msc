import tensorflow as tf
import re

def get_tf_dataset(mode):
    if (mode=='train'):
        dataset = tf.data.TextLineDataset('datasets/data/Dong2014/train.txt')
    elif (mode=='test'):
        dataset = tf.data.TextLineDataset('datasets/data/Dong2014/test.txt')
    else:
        raise ValueError('mode has to be train or test')

    labels = (dataset
                .filter(lambda line: tf.strings.regex_full_match(line, r"^[-10]*$"))
                .map(lambda label: tf.string_to_number(label)))

    sentences = (dataset
                    .filter(lambda line:  tf.strings.regex_full_match(line, r".*\$T\$.*")))
    
    targets = (dataset
                .filter(lambda line: tf.equal(tf.size(tf.string_split([line])),tf.size(tf.string_split([tf.strings.regex_replace(line, r"\$T\$", "")])))))

    return tf.data.Dataset.zip((labels, (sentences, targets)))

def prep_for_input_fn(mode='test'):
    if (mode=='train'):
        file_path = 'datasets/data/Dong2014/train.txt'
    else:
        file_path = 'datasets/data/Dong2014/test.txt'

    features = {
        'sentence' : [],
        'sentence_length': [],
        'target' : [] 
    }
    labels = []
    
    with open(file_path, "r") as f:
        for line in f:
            if '$T$' in line:
                features['sentence'].append(line.strip())
                features['sentence_length'].append(len(line.strip()))
            elif '$T$' not in line and not(re.match(r"^[-10]*$", line)):
                features['target'].append(line.strip())
            elif re.match(r"^[-10]*$", line.strip()):
                labels.append(int(line.strip()))

    return features, labels


