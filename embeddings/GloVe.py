from embeddings.Embedding import Embedding

class GloVe(Embedding):

    def __init__(self, alias, version):
        path = 'GloVe/'
        if alias == 'twitter':
            if version == 'Debug':
                path += 'glove.'+alias+'.27B/glove.'+alias+'.'+version+'.25d.txt'
            else:
                path += 'glove.'+alias+'.27B/glove.'+alias+'.27B.'+version+'d.txt'
        else:
            path += 'glove.'+alias+'/glove.'+alias+'.'+version+'d.txt'

        super().__init__(path=path, alias=alias, version=version)