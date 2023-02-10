import argparse
from typing import Tuple
from unittest.result import failfast
import scipy.sparse as sp
import numpy as np
import random as rd
import itertools
from random_choice import randint_choice
from sklearn.model_selection import train_test_split

from util.setting import log
from util.load_data import Data

class GNNLoader(Data):
    def __init__(self, args:argparse.Namespace) -> None:
        super().__init__(args)

        self.adj_type = args.adj_type

        self.curr_clone_type = -1

        # generate sparse adjacency matrices for system entity inter_data
        log.info("Converting interactions into sparse adjacency matrix")
        adj_list = self._get_relational_adj_list(self.inter_data)

        # generate normalized (sparse adjacency) metrices
        log.info("Generating normalized sparse adjacency matrix")
        self.norm_list = self._get_relational_norm_list(adj_list)

        # load the norm matrix (used for information propagation)
        self.A_in = sum(self.norm_list)

        # mess_dropout
        self.mess_dropout = eval(args.mess_dropout)

        # split functions into training/validation/testing sets for code clone
        if args.clone_test_supervised:
            log.info('Generating code clone training, validation, and testing sets')
            self.clone_test_size = args.clone_test_size
            self.clone_val_size = args.clone_val_size
            if args.dataset.find('bcb') == -1:
                self.clone_train_data, self.clone_val_data, self.clone_test_data = self._sample_clone_oj_split()
                self.n_clone_train, self.n_clone_val, self.n_clone_test_supervised = len(self.clone_train_data), len(self.clone_val_data), len(self.clone_test_data)
                # batch iter
                self.clone_data_iter = self.n_clone_train // self.batch_size_clone
                if self.n_clone_train % self.batch_size_clone:
                    self.clone_data_iter += 1

        # sample positive/negative paris from code clone dataset
        if args.clone_test_unsupervised:
            log.info('Sampling positive and negative pairs for code clone')
            self.clone_pos_pairs, self.clone_neg_pairs = self._sample_clone_pair()
            self.n_clone_test_unsupervised = len(self.clone_pos_pairs)

        # split functions into training/validation/testing sets for code classification
        if args.classification_test:
            log.info('Generating code classification training, validation, and testing sets')
            self.class_test_size = args.class_test_size
            self.class_val_size = args.class_val_size
            self.classification_train_data, self.classification_val_data, self.classification_test_data = self._sample_classification_split()
            self.n_classification_train, self.n_classification_val, self.n_classification_test = len(self.classification_train_data), len(self.classification_val_data), len(self.classification_test_data)
            # batch iter
            # Todo:
            self.classification_data_iter = self.n_classification_train // self.batch_size_classification
            if self.n_classification_train % self.batch_size_classification:
                self.classification_data_iter += 1

        # sample functions from code clone dataset
        if args.cluster_test:
            log.info('Sampling functions for code cluster')
            self.cluster_test_data = self._sample_cluster()
            self.n_cluster_test = len(self.cluster_test_data)

    def _get_relational_adj_list(self, inter_data) -> Tuple[list, list]:
        def _np_mat2sp_adj(np_mat:np.array, row_pre=0, col_pre=0) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
            # to-node interaction: A: A->B
            a_rows = np_mat[:, 0] + row_pre # all As
            a_cols = np_mat[:, 1] + col_pre # all Bs
            # must use float 1. (int is not allowed)
            a_vals = [1.] * len(a_rows)

            # from-node interaction: A: B->A
            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            # self.n_entity + 1: 
            # we add a `ghost` entity to support parallel AST node embedding 
            # retrival for program statements
            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(self.n_entity + 1, self.n_entity + 1))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(self.n_entity + 1, self.n_entity + 1))

            return a_adj, b_adj

        adj_mat_list = []

        r, r_inv = _np_mat2sp_adj(inter_data)
        adj_mat_list.append(r)
        # Todo: whether r_inv (inverse directions) helps infer code representations
        adj_mat_list.append(r_inv)

        return adj_mat_list
    
    def _get_relational_norm_list(self, adj_list:str) -> list:
        # Init for 1/Nt
        def _si_norm(adj):
            rowsum = np.array(adj.sum(axis=1))
            # np.power(rowsum, -1).flatten() may trigger divide by zero
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj

        # Init for 1/(Nt*Nh)^(1/2)
        def _bi_norm(adj):
            rowsum = np.array(adj.sum(axis=1))
            # np.power(rowsum, -1).flatten() may trigger divide by zero
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            # Different from KGAT's implementation
            # bi_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_norm

        if self.adj_type == 'bi':
            norm_list = [_bi_norm(adj) for adj in adj_list]
        else:
            norm_list = [_si_norm(adj) for adj in adj_list]

        return norm_list

    def _sample_cpg_split(self, inter_data:list) -> Tuple[list, list, list]:
        # we use the whole dataset to pretrain the model
        inter_train_data = inter_data

        _, inter_test_val_data = train_test_split(
            inter_data, test_size=self.ssl_test_size + self.ssl_val_size,
            random_state=2022)
        
        inter_val_data, inter_test_data = train_test_split(
            inter_test_val_data, test_size=self.ssl_test_size / (self.ssl_test_size + self.ssl_val_size),
            random_state=2022)

        return inter_train_data, inter_val_data, inter_test_data

    def sample_pos_pair(self, l, num):
        """randomly sample num (e.g., 100) non-repeated pairs from list (l) """
        pairs = itertools.combinations(l, 2)
        pool = tuple(pairs)
        n = len(pool)
        indices = sorted(rd.sample(range(n), num))
        return tuple(pool[i] for i in indices)

    def sample_neg_pair(self, l1, l2, num):
        "Random selection from itertools.product(*args, **kwds)"
        pairs = itertools.product(l1, l2)
        pool = tuple(pairs)
        n = len(pool)
        indices = sorted(rd.sample(range(n), num))
        return tuple(pool[i] for i in indices)

    def _sample_clone_pair(self, sample_clone_num=20000):
        """"""
        clone_pos_pairs = []
        clone_neg_pairs = []

        all_clone_family = self.all_clone_family

        exist_family_pair = [-1 for _ in range(self.classification_num)]
        for idx, family in enumerate(all_clone_family):
            while True:
                idx_neg = randint_choice(self.classification_num, size=1, replace=False)
                if idx_neg != idx and exist_family_pair[idx_neg] != idx:
                    exist_family_pair[idx] = idx_neg
                    break

            clone_pos_pairs_family = self.sample_pos_pair(family, sample_clone_num)
            clone_neg_pairs_family = self.sample_neg_pair(family, all_clone_family[idx_neg], sample_clone_num)
            
            clone_pos_pairs.extend(clone_pos_pairs_family)
            clone_neg_pairs.extend(clone_neg_pairs_family)
        
        log.debug('Code clone: #positive pairs {}'.format(len(clone_pos_pairs)))
        log.debug('Code clone: #negative pairs {}'.format(len(clone_neg_pairs)))

        assert(len(clone_pos_pairs) == len(clone_neg_pairs))

        return clone_pos_pairs, clone_neg_pairs  
    
    def _sample_cluster(self):
        """"""
        cluster_test_data = []

        all_clone_family = self.all_clone_family
        for idx, family in enumerate(all_clone_family):
            cluster_test_data.extend([[func, idx] for func in family])
            
        log.debug('Code Cluster: #samples {}'.format(len(cluster_test_data)))

        return cluster_test_data

    def _sample_classification_split(self):
        """"""
        class_train_data, class_val_data, class_test_data = [], [], []
        
        all_clone_family = self.all_clone_family

        for idx, family in enumerate(all_clone_family):
            class_train_family, class_test_val_family = train_test_split(
                family, test_size=self.class_test_size+self.class_val_size,
                random_state=2022)
        
            class_val_family, class_test_family = train_test_split(
                class_test_val_family, test_size=self.class_test_size/(self.class_test_size+self.class_val_size),
                random_state=2022)

            class_train_data.extend([[func, idx] for func in class_train_family])
            class_val_data.extend([[func, idx] for func in class_val_family])
            class_test_data.extend([[func, idx] for func in class_test_family])
        
        log.debug('Code Classification [n_train, n_val, n_test] = [%d, %d, %d]' % (len(class_train_data), len(class_val_data), len(class_test_data)))

        if self.split_label:
            f = open('oj_classification_split.txt', 'w')

            # training/validation/test: 0/1/2
            for i in range(len(class_train_data)):
                f.write("%d %d %d\n" % (class_train_data[i][0][0], class_train_data[i][1], 0))

            for i in range(len(class_val_data)):
                f.write("%d %d %d\n" % (class_val_data[i][0][0], class_val_data[i][1], 1))

            for i in range(len(class_test_data)):
                f.write("%d %d %d\n" % (class_test_data[i][0][0], class_test_data[i][1], 2))

            f.close()

        return class_train_data, class_val_data, class_test_data

    def _generate_clone_bcb_split(self, curr_clone_type):
        self.curr_clone_type = curr_clone_type
        self.clone_train_data, self.clone_val_data, self.clone_test_data = self._sample_clone_bcb_split()
        self.n_clone_train, self.n_clone_val, self.n_clone_test_supervised = len(self.clone_train_data), len(self.clone_val_data), len(self.clone_test_data)
        # batch iter
        self.clone_data_iter = self.n_clone_train // self.batch_size_clone
        if self.n_clone_train % self.batch_size_clone:
            self.clone_data_iter += 1

    def _sample_clone_bcb_split(self):
        clone_train_data, clone_val_data, clone_test_data = [], [], []
        all_clone_type = self.all_clone_type
        for idx, clone_type in enumerate(all_clone_type):
            # idx == 5 represents negative samples
            if (idx == 5 or idx == self.curr_clone_type):
                clone_train_family, clone_test_val_family = train_test_split(
                    clone_type, test_size=self.clone_test_size+self.clone_val_size,
                    random_state=999)
                
                clone_val_family, clone_test_family = train_test_split(
                    clone_test_val_family, test_size=self.clone_test_size/(self.clone_test_size+self.clone_val_size),
                    random_state=999)

                clone_train_data.extend([[pair, [idx]] for pair in clone_train_family])
                clone_val_data.extend([[pair, [idx]] for pair in clone_val_family])
                clone_test_data.extend([[pair, [idx]] for pair in clone_test_family])

        log.debug('Code Clone (Type %d) [n_train, n_val, n_test] = [%d, %d, %d]' % (self.curr_clone_type, len(clone_train_data), len(clone_val_data) , len(clone_test_data)))

        if self.split_label:
            f = open('bcb_clone_split.txt', 'w')

            # training/validation/test: 0/1/2
            for i in range(len(clone_train_data)):
                f.write("%d %d %d %d\n" % (clone_train_data[i][0][0][0], clone_train_data[i][0][1][0], clone_train_data[i][1][0], 0))

            for i in range(len(clone_val_data)):
                f.write("%d %d %d %d\n" % (clone_val_data[i][0][0][0], clone_val_data[i][0][1][0], clone_val_data[i][1][0], 1))

            for i in range(len(clone_test_data)):
                f.write("%d %d %d %d\n" % (clone_test_data[i][0][0][0], clone_test_data[i][0][1][0], clone_test_data[i][1][0], 2))

            f.close()
        else:
            # Identify positives and negatives from different clone types
            for i in range(len(clone_train_data)):
                if clone_train_data[i][1][0] == 5:
                    # True Negatives
                    clone_train_data[i][1][0] = 0
                else:
                    # True Positive
                    clone_train_data[i][1][0] = 1

            for i in range(len(clone_val_data)):
                if clone_val_data[i][1][0] == 5:
                    # True Negatives
                    clone_val_data[i][1][0] = 0
                else:
                    # True Positive
                    clone_val_data[i][1][0] = 1

            for i in range(len(clone_test_data)):
                if clone_test_data[i][1][0] == 5:
                    # True Negatives
                    clone_test_data[i][1][0] = 0
                else:
                    # True Positive
                    clone_test_data[i][1][0] = 1

        return clone_train_data, clone_val_data, clone_test_data

    def _sample_clone_oj_split(self, sample_clone_num=20000):
        """"""
        clone_train_data, clone_val_data, clone_test_data = [], [], []

        all_clone_family = self.all_clone_family

        exist_family_pair = [-1 for _ in range(self.classification_num)]
        for idx, family in enumerate(all_clone_family):
            while True:
                idx_neg = randint_choice(self.classification_num, size=1, replace=False)
                if idx_neg != idx and exist_family_pair[idx_neg] != idx:
                    exist_family_pair[idx] = idx_neg
                    break

            # we follow the data distribution in ASTNN to generate our
            # clone/non-clone 
            clone_pos_pairs_family = self.sample_pos_pair(family, int(sample_clone_num * 0.066))
            clone_neg_pairs_family = self.sample_neg_pair(family, all_clone_family[idx_neg], sample_clone_num)
            
            # split positive code clone pairs
            clone_train_family, clone_test_val_family = train_test_split(
                clone_pos_pairs_family, test_size=self.clone_test_size+self.clone_val_size,
                random_state=2022)
            
            clone_val_family, clone_test_family = train_test_split(
                clone_test_val_family, test_size=self.clone_test_size/(self.clone_test_size+self.clone_val_size),
                random_state=2022)

            clone_train_data.extend([[pair, [1]] for pair in clone_train_family])
            clone_val_data.extend([[pair, [1]] for pair in clone_val_family])
            clone_test_data.extend([[pair, [1]] for pair in clone_test_family])

            # split negative code clone pairs
            clone_train_family, clone_test_val_family = train_test_split(
                clone_neg_pairs_family, test_size=self.clone_test_size+self.clone_val_size,
                random_state=2022)
            
            clone_val_family, clone_test_family = train_test_split(
                clone_test_val_family, test_size=self.clone_test_size/(self.clone_test_size+self.clone_val_size),
                random_state=2022)

            clone_train_data.extend([[pair, [0]] for pair in clone_train_family])
            clone_val_data.extend([[pair, [0]] for pair in clone_val_family])
            clone_test_data.extend([[pair, [0]] for pair in clone_test_family])

        log.debug('Code Clone [n_train, n_val, n_test] = [%d, %d, %d]' % (len(clone_train_data), len(clone_val_data), len(clone_test_data)))

        if self.split_label:
            f = open('oj_clone_split.txt', 'w')

            # training/validation/test: 0/1/2
            for i in range(len(clone_train_data)):
                # pair[0] represent ojclone file_id
                f.write("%d %d %d %d\n" % (clone_train_data[i][0][0][0], clone_train_data[i][0][1][0], clone_train_data[i][1][0], 0))

            for i in range(len(clone_val_data)):
                f.write("%d %d %d %d\n" % (clone_val_data[i][0][0][0], clone_val_data[i][0][1][0], clone_val_data[i][1][0], 1))

            for i in range(len(clone_test_data)):
                f.write("%d %d %d %d\n" % (clone_test_data[i][0][0][0], clone_test_data[i][0][1][0], clone_test_data[i][1][0], 2))
            f.close()
            exit(-1)

        return clone_train_data, clone_val_data, clone_test_data

    def transfer_s_to_e(self, s_list:list) -> np.array:
        """"""
        s2e = []
        for s in s_list:
            s2e.append(self.stat_dict[s])
        return np.array(s2e)

    def _convert_csr_to_sparse_tensor_inputs(self, X:sp.csr_matrix) -> Tuple[list, list, list]:
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

    def extract_stat_from_inter(self, inter_data:list) -> Tuple[list, list]:
        s1 = []
        s2 = []
        
        for pair in inter_data:
            s1.append(pair[0])
            s2.append(pair[1])

        return s1, s2

    def generate_clone_train_batch(self, i_batch:int) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_clone
        if i_batch == self.clone_data_iter - 1:
            end = self.n_clone_train
        else:
            end = (i_batch + 1) * self.batch_size_clone
        
        f_y = self.clone_train_data[start: end]

        f1_clone = [f[0][0] for f in f_y]
        f2_clone = [f[0][1] for f in f_y]
        y_clone = [f[1] for f in f_y]

        batch_data['f1_clone'] = f1_clone
        batch_data['f2_clone'] = f2_clone
        batch_data['y_clone'] = y_clone

        return batch_data

    def generate_clone_val_batch(self, i_batch:int, last_batch:bool) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_clone
        if last_batch:
            end = self.n_clone_val
        else:
            end = (i_batch + 1) * self.batch_size_clone

        f_y = self.clone_val_data[start: end]

        f1_clone = [f[0][0] for f in f_y]
        f2_clone = [f[0][1] for f in f_y]
        y_clone = [f[1] for f in f_y]

        batch_data['f1_clone'] = f1_clone
        batch_data['f2_clone'] = f2_clone
        batch_data['y_clone'] = y_clone

        return batch_data

    def generate_clone_test_batch(self, i_batch:int, last_batch:bool) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_clone
        if last_batch:
            end = self.n_clone_test_supervised
        else:
            end = (i_batch + 1) * self.batch_size_clone

        f_y = self.clone_test_data[start: end]

        f1_clone = [f[0][0] for f in f_y]
        f2_clone = [f[0][1] for f in f_y]
        y_clone = [f[1] for f in f_y]

        batch_data['f1_clone'] = f1_clone
        batch_data['f2_clone'] = f2_clone
        batch_data['y_clone'] = y_clone

        return batch_data

    def generate_clone_train_feed_dict(self, model, batch_data):
        feed_dict = {
            # Clone
            model.f1_clone: batch_data['f1_clone'],
            model.f2_clone: batch_data['f2_clone'],
            model.y_clone: batch_data['y_clone'],
            # hardcode dropping probability as 0
            model.mess_dropout: [0,0,0,0,0,0]
        }
        return feed_dict

    def generate_classification_train_batch(self, i_batch:int) -> dict:
        """"""
        batch_data = {}

        start = i_batch * self.batch_size_classification
        if i_batch == self.classification_data_iter:
            end = self.n_classification_train
        else:
            end = (i_batch + 1) * self.batch_size_classification

        f_y = self.classification_train_data[start: end]

        batch_data['f_classification'] = [f[0] for f in f_y]
        # labels, e.g., 2
        batch_data['y_classification'] = [f[1] for f in f_y]
        batch_data['y_classification'] = np.array(batch_data['y_classification']).flatten()

        return batch_data

    def generate_classification_val_batch(self, i_batch:int, last_batch:bool) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_classification
        if last_batch:
            end = self.n_classification_val
        else:
            end = (i_batch + 1) * self.batch_size_classification

        f_y = self.classification_val_data[start: end]

        batch_data['f_classification'] = [f[0] for f in f_y]
        # labels, e.g., 2
        batch_data['y_classification'] = [f[1] for f in f_y]

        return batch_data

    def generate_classification_test_batch(self, i_batch:int, last_batch:bool) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_classification
        if last_batch:
            end = self.n_classification_test
        else:
            end = (i_batch + 1) * self.batch_size_classification

        f_y = self.classification_test_data[start: end]

        batch_data['f_classification'] = [f[0] for f in f_y]
        # labels, e.g., 2
        batch_data['y_classification'] = [f[1] for f in f_y]

        return batch_data

    def generate_classification_train_feed_dict(self, model, batch_data):
        feed_dict = {
            # Classification
            model.f_classification: batch_data['f_classification'],
            model.y_classification: batch_data['y_classification'],
            # hardcode dropping probability as 0
            model.mess_dropout: [0,0,0,0,0,0],
        }
        
        return feed_dict

    def generate_classification_val_feed_dict(self, model, batch_data):
        feed_dict = {
            model.f_classification: batch_data['f_classification'],
            # hardcode dropping probability as 0
            model.mess_dropout: [0,0,0,0,0,0],
        }
        return feed_dict

    def generate_clone_batch(self, i_batch:int, last_batch:bool, pos:bool) -> dict:
        batch_data = {}

        start = i_batch * self.batch_size_clone
        if last_batch:
            end = self.n_clone_test_unsupervised
        else:
            end = (i_batch + 1) * self.batch_size_clone
        
        if pos:
            f1_clone = [pair[0] for pair in self.clone_pos_pairs[start:end]]
            f2_clone = [pair[1] for pair in self.clone_pos_pairs[start:end]]
        else:
            f1_clone = [pair[0] for pair in self.clone_neg_pairs[start:end]]
            f2_clone = [pair[1] for pair in self.clone_neg_pairs[start:end]]

        batch_data['f1_clone'] = f1_clone
        batch_data['f2_clone'] = f2_clone

        return batch_data

    def generate_clone_feed_dict(self, model, batch_data):
        feed_dict = {
            model.f1_clone: batch_data['f1_clone'],
            model.f2_clone: batch_data['f2_clone'],
            # hardcode dropping probability as 0
            model.mess_dropout: [0,0,0,0,0,0],
        }
        return feed_dict

    def generate_cluster_batch(self, i_batch:int, last_batch:bool) -> dict:
        batch_data = {}
        
        start = i_batch * self.batch_size_cluster
        if last_batch:
            end = self.n_cluster_test
        else:
            end = (i_batch + 1) * self.batch_size_cluster

        f_y= self.cluster_test_data[start:end]

        batch_data['f_cluster'] = [f[0] for f in f_y]
        batch_data['cluster_label'] = [f[1] for f in f_y]

        return batch_data
    
    def generate_cluster_feed_dict(self, model, batch_data):
        feed_dict = {
            model.f_cluster: batch_data['f_cluster'],
            # hardcode dropping probability as 0
            model.mess_dropout: [0,0,0,0,0,0],
        }
        return feed_dict
