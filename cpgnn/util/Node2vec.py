from gensim.models import word2vec
import numpy as np
import argparse
import networkx as nx
from node2vec import Node2Vec

from util.helper import ensure_dir
from util.setting import log

class Node2vec:
    def __init__(self, args:argparse.Namespace, inter_data) -> None:
        self.nx_G = self.read_graph(inter_data)
        self.walk_length = args.walk_length
        self.num_walks = args.num_walks

        self.type_dim = args.type_dim
        self.window = args.word2vec_window
        self.min_count = args.word2vec_count
        self.worker = args.word2vec_worker
        self.model = None
        self.embedding = None

    def read_graph(self, inter_data):
        '''
        Reads the input network in networkx.
        '''
        G = nx.Graph()
        G.add_edges_from(inter_data)

        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        return G

    def init_embedding(self, e2t_list:list, typetoken_seq:list):
        """"""
        log.info('Initing type/token embeddings with node2vec')
        node2vec = Node2Vec(self.nx_G, 
            dimensions=self.type_dim, walk_length=self.walk_length, 
            num_walks=self.num_walks, workers=self.worker, seed=2022)
        
        model = node2vec.fit(window=self.window, min_count=self.min_count, batch_words=4)

        init_embedding = np.zeros(shape=(len(e2t_list), 2*self.type_dim), dtype=np.float32)
        
        # Embedding: entity = type || token
        for idx, typetoken in enumerate(e2t_list):
            if typetoken[1] != -1:
                init_embedding[idx] = np.append(model.wv[typetoken[0]], model.wv[typetoken[1]])
            else:
                init_embedding[idx] = np.append(model.wv[typetoken[0]], np.zeros(self.type_dim))

        self.embedding = init_embedding
        self.model = model

    def store_embedding(self, pretrain_path:str):
        """"""
        embedding_save_dir = '%s/node2vec/%s_%s_%s/' % \
            (pretrain_path, self.type_dim, self.window, self.min_count)
        ensure_dir(embedding_save_dir)

        embedding_path = embedding_save_dir + 'node2vec.embedding'

        with open (embedding_path, 'wb') as f:
            np.save(f, self.embedding)

        log.info("save node2vec embeddings in %s" % embedding_path)

    def load_embedding(self, pretrain_path:str):
        """"""
        embedding_path = '%s/node2vec/%s_%s_%s/node2vec.embedding' % \
            (pretrain_path, self.type_dim, self.window, self.min_count)

        with open(embedding_path, 'rb') as f:
            self.embedding = np.load(embedding_path)

        log.info("load node2vec embeddings in %s" % embedding_path)
