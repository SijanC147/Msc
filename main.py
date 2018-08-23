import tensorflow as tf

from datasets.Dong2014 import dong2014

dataset = dong2014.get_raw_dataset(mode='test')

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()

print(sess.run(next_element))

sess.close()
