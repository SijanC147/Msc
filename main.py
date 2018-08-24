import tensorflow as tf

from datasets import dong2014
from embeddings.GloVe import GloVe 

dataset = dong2014.get_raw_dataset(mode='test')
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

myGloveTest = GloVe('twitterMicro')
phrases = ['Hello my name is Sean.', 'The iPhone has a great display, samsung can suck a dick, their battery explodes❗️', 'I love deep learning ♥']

embedding_matrix = tf.constant(myGloveTest.get_embedding_vectors(), name='embedding_matrix')
x = tf.placeholder(dtype=tf.int64)

embed = tf.nn.embedding_lookup(embedding_matrix, x, name='embedding_layer')

sess = tf.Session()
for mapped_ids in myGloveTest.map_embedding_ids(phrases=phrases):

    print(sess.run(embed, feed_dict={x: mapped_ids}))
# print(sess.run(next_element))
sess.close()
