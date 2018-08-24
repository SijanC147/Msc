from embeddings.Embedding import Embedding

class GloVe(Embedding):

    def __init__(self, version):
        path = 'GloVe/'
        if version == 'twitterMicro':
            path += 'glove.twitter.27B/glove.twitter.Micro.25d.txt'

        super().__init__(embedding_path=path)