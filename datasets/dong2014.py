import tensorflow as tf

def get_raw_dataset(mode):
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