from gensim.models import word2vec
import numpy as np
import argparse

from util.helper import ensure_dir
from util.setting import log

class Word2vec:
    def __init__(self, args:argparse.Namespace) -> None:
        self.type_dim = args.type_dim
        self.window = args.word2vec_window
        self.min_count = args.word2vec_count
        self.worker = args.word2vec_worker
        self.model = None
        self.embedding = None
        self.embed_init = args.embed_init

    def init_embedding(self, e2t_list:list, typetoken_seq:list):
        """"""
        log.info('Initing type/token embeddings with word2vec')
        model = word2vec.Word2Vec(sentences=typetoken_seq, 
            vector_size=self.type_dim, window=self.window, 
            min_count=self.min_count, workers=self.worker, seed=2022)

        init_embedding = np.zeros(shape=(len(e2t_list), 2*self.type_dim), dtype=np.float32)
        
        # Combined entity = type || token
        if self.embed_init == 'comb':
            for idx, typetoken in enumerate(e2t_list):
                if typetoken[1] != -1:
                    init_embedding[idx] = np.append(model.wv[typetoken[0]], model.wv[typetoken[1]])
                else:
                    init_embedding[idx] = np.append(model.wv[typetoken[0]], np.zeros(self.type_dim))
        # Type entity = type || 0
        elif self.embed_init == 'type':
            for idx, typetoken in enumerate(e2t_list):
                init_embedding[idx] = np.append(model.wv[typetoken[0]], np.zeros(self.type_dim))
        # Type entity = 0 || token
        elif self.embed_init == 'token':
            for idx, typetoken in enumerate(e2t_list):
                if typetoken[1] != -1:
                    init_embedding[idx] = np.append(np.zeros(self.type_dim), model.wv[typetoken[1]])
                else:
                    init_embedding[idx] = np.append(np.zeros(self.type_dim), np.zeros(self.type_dim))
        else:
            log.error('unknown embedding')
            exit(-1)

        self.embedding = init_embedding
        self.model = model

    def store_embedding(self, pretrain_path:str):
        """"""
        embedding_save_dir = '%s/word2vec/%s_%s_%s/' % \
            (pretrain_path, self.type_dim, self.window, self.min_count)
        ensure_dir(embedding_save_dir)

        embedding_path = embedding_save_dir + 'word2vec.embedding'

        with open (embedding_path, 'wb') as f:
            np.save(f, self.embedding)

        log.info("save word2vec embeddings in %s" % embedding_path)

    def load_embedding(self, pretrain_path:str):
        """"""
        embedding_path = '%s/word2vec/%s_%s_%s/word2vec.embedding' % \
            (pretrain_path, self.type_dim, self.window, self.min_count)

        with open(embedding_path, 'rb') as f:
            self.embedding = np.load(embedding_path)

        log.info("load word2vec embeddings in %s" % embedding_path)

    def print_embedding(self):
        log.debug(self.embedding)
