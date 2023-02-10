import argparse
from typing import Tuple
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import os

from util.helper import log
from model.load_gnn import GNNLoader
from util.helper import ensure_dir

class SGL(object):
    def __init__(self, args:argparse.Namespace, data_generator:GNNLoader,
    pretrain_embedding:np.array) -> None:
        super().__init__()
        """ Parse arguments for SGL """
        self._parse_args(args, data_generator, pretrain_embedding)

        """ Create placeholder for training inputs """
        self._build_inputs()

        """ Create variable for training weights """
        with tf.device("/device:GPU:" + args.gpu_id[0]):
            self._build_weights()
        
        """ Compute code representation via GNN """
        with tf.device("/device:GPU:" + args.gpu_id[0]):
            self._build_inter_model()

        """ Build model/loss for code classification """
        with tf.device("/device:GPU:" + args.gpu_id[1]):
            if self.classification_train:
                self._build_classification_model()
                self._build_classification_loss()

        """ Build model for code clone """
        with tf.device("/device:GPU:" + args.gpu_id[1]):
            if self.clone_train:
                self._build_clone_model()
                self._build_clone_loss()

        """ Build model for code cluster """
        # self._build_cluster_model()

        """ Count the number of model parameters """
        self._statistics_params()

    def setup_sess(self) -> tf.Session:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # tf_config.log_device_placement = True
        sess = tf.Session(config=tf_config)
        sess.run(tf.global_variables_initializer())
        return sess

    def _parse_args(self, args:argparse.Namespace, data_generator:GNNLoader, 
    pretrain_embedding:np.array) -> None:
        """"""
        # pretrain word2vec embedding
        self.pretrain_embedding = pretrain_embedding

        # learning flags
        self.clone_train = args.clone_test_supervised
        self.classification_train = args.classification_test

        # statistics for dataset 
        self.n_entity = data_generator.n_entity
        self.max_entity_stat = data_generator.max_entity_stat
        self.max_entity_func = data_generator.max_entity_func

        # setting for gnn
        self.entity_dim = args.type_dim * 2
        self.weight_size = eval(args.layer_size)
        self.n_layer = len(self.weight_size)
        self.agg_type = args.agg_type
        if self.agg_type in ['none']:
            self.inter_dim = self.entity_dim
        else:
            self.inter_dim = self.entity_dim + sum(self.weight_size) # sum -> gnn aggregation
        # assert(self.entity_dim == self.weight_size[0])

        # setting for loss function (optimization)
        self.regs = eval(args.regs)
        self.opt_type = args.opt_type
        self.lr = args.lr

        # init adjacency matrix for gnn (setup n_fold to avoid 'run out of
        # memory' during model training)
        self.A_in = data_generator.A_in
        self.n_fold = 1

        self.model_type = '%s_%s_%s' % (args.model_type, args.adj_type, self.agg_type)
        self.layer_type = '-'.join([str(l) for l in eval(args.layer_size)])
        self.reg_type = '-'.join([str(r) for r in eval(args.regs)])
        
        # setting for code classification
        self.classification_num = data_generator.classification_num

    def _build_inputs(self) -> None:
        """"""
        # dropout: message dropout (adopted on GNN aggregation)
        self.mess_dropout =tf.placeholder(tf.float32, shape=[None], name='mess_dropout')
        
        # input function/label for code classification
        self.f_classification = tf.placeholder(dtype=tf.int64, name='f_classification',
            shape=[None, self.max_entity_func])
        self.y_classification = tf.placeholder(dtype=tf.int64, name='y_classification',
            shape=[None])

        # input a pair of functions for code clone
        # f_e shape: [batch_size, entity_num]
        self.f1_clone = tf.placeholder(dtype=tf.int64, name='f1_clone',
            shape=[None, self.max_entity_func])
        self.f2_clone = tf.placeholder(dtype=tf.int64, name='f2_clone',
            shape=[None, self.max_entity_func])
        self.y_clone = tf.placeholder(dtype=tf.float32, name='y_clone',
            shape=[None, 1])

        # input function for code cluster
        self.f_cluster = tf.placeholder(dtype=tf.int64, name='f_cluster',
            shape=[None, self.max_entity_func])

        log.info("Finish building inputs for SGL")

    def _build_weights(self) -> None:
        """"""
        all_weight = dict()
        initializer = tf.contrib.layers.xavier_initializer(seed=2022)
    
        # weights for entity embeddings: fine-tune does not alter entity embeddings
        entity_trainable = False

        if self.pretrain_embedding is None:
            all_weight['entity_embedding'] = tf.Variable(
            initial_value=initializer([self.n_entity, self.entity_dim]),
            dtype=tf.float32,
            trainable=entity_trainable,
            name='entity_embedding')
            log.info("Init entity embeddings with Xavier")
        else:
            all_weight['entity_embedding'] = tf.Variable(
            initial_value=self.pretrain_embedding,
            trainable=entity_trainable,
            name='entity_embedding')
            log.info("Init entity embeddings with pre-trained word2vec embeddings")

        # we add a `ghost` entity whose embedding is [0,..,0] to allow using 
        # tf.nn.embedding_lookup for obtaining program statement representations
        paddings = tf.constant([[0, 1], [0, 0]])
        all_weight['entity_embedding'] = tf.pad(
            all_weight['entity_embedding'], paddings, "CONSTANT", 
            name='entity_embedding_pad')

        weight_size_list = [self.entity_dim] + self.weight_size

        # weights for gnn
        for k in range(self.n_layer):
            if self.agg_type in ['gcn']:
                all_weight['w_gcn_%d' % k] = tf.Variable(
                    initial_value=initializer([weight_size_list[k], weight_size_list[k + 1]]),
                    name='w_gcn_%d' % k)
                all_weight['b_gcn_%d' %k] = tf.Variable(
                    initial_value=initializer([1, weight_size_list[k + 1]]),
                    name='b_gcn_%d' % k)
            
            elif self.agg_type in ['gnn']:
                all_weight['w_gnn_%d' % k] = tf.Variable(
                    initial_value=initializer([weight_size_list[k] * 2, weight_size_list[k + 1]]),
                    name='w_gnn_%d' % k)
                all_weight['b_gnn_%d' % k] = tf.Variable(
                    initial_value=initializer([1, weight_size_list[k + 1]]),
                    name='b_gnn_%d' % k)    

            elif self.agg_type in ['ggnn']:
                all_weight['ggnn_gru'] = tf.contrib.rnn.GRUCell(self.weight_size[0])
                for size in self.weight_size:
                    assert(size == self.weight_size[0])
                break
                
            elif self.agg_type in ['none', 'lightgcn']:
                pass
            
            elif self.agg_type in ['kgat']:
                all_weight['w1_kgat_%d' % k] = tf.Variable(
                    initial_value=initializer([weight_size_list[k], weight_size_list[k + 1]]),
                    name='w1_kgat_%d' % k)
                all_weight['b1_kgat_%d' % k] = tf.Variable(
                    initial_value=initializer([1, weight_size_list[k + 1]]),
                    name='b1_kgat_%d' % k)
                all_weight['w2_kgat_%d' % k] = tf.Variable(
                    initial_value=initializer([weight_size_list[k], weight_size_list[k + 1]]),
                    name='w2_kgat_%d' % k)
                all_weight['b2_kgat_%d' % k] = tf.Variable(
                    initial_value=initializer([1, weight_size_list[k + 1]]),
                    name='b2_kgat_%d' % k)

            else:
                log.error('aggregator type for GNN is unknown')
                exit(-1)

        # weights for code classification
        if self.classification_train:
            all_weight['w_classification'] = tf.Variable(
                initial_value=initializer([self.inter_dim, self.classification_num]), name='w_classification')
            all_weight['b_classification'] = tf.Variable(
                initial_value=initializer([1, self.classification_num]), name='b_classification')  

        # weights for code clone
        if self.clone_train:
            all_weight['w_clone'] = tf.Variable(
                initial_value=initializer([self.inter_dim, 1]), name='w_clone')
            all_weight['b_clone'] = tf.Variable(
                initial_value=initializer([1, 1]), name='b_clone')

        self.weights = all_weight
        
        log.info("Finish building weights for SGL")

    def _build_inter_model(self):
        """ Different Convolutional Layer:
        1. gcn: 'Semi-Supervised Classification with Graph Convolutional
           Networks',ICLR'18
        2. lightgcn: 'LightGCN: Simplifying and Powering Graph Convolution
           Network for Recommendation', SIGIR'20
        3. ggnn: 'Gated graph sequence neural networks', ICLR'16. """
        if self.agg_type in ['gnn']:
            self.g_embedding = self._create_gnn_embed()
        elif self.agg_type in ['gcn']:
            self.g_embedding = self._create_gcn_embed()
        elif self.agg_type in ['lightgcn']:
            self.g_embedding = self._create_lightgcn_embed()
        elif self.agg_type in ['ggnn']:
            self.g_embedding = self._create_ggnn_embed()
        elif self.agg_type in ['kgat']:
            self.g_embedding = self._create_kgat_embed()
        elif self.agg_type in ['none']:
            log.warn("No use of GNN")
            self.g_embedding = self.weights['entity_embedding']
        else:
            log.error('aggregator type for GNN is unknown')
            exit(-1)    

        log.info('Finish building model for GNN')

    def _split_A_hat(self, A:sp.coo_matrix) -> list:
        """"""
        A_fold_hat = []
        fold_len = self.n_entity // self.n_fold

        A = A.tocsr()

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_entity + 1
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(A[start:end]))

        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X:sp.csr_matrix) -> tf.SparseTensor:
        """"""
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()

        if len(coo.data) > 0:
            return tf.SparseTensor(indices, coo.data, coo.shape)
        else:
            return tf.SparseTensor(indices=np.empty((0,2), dtype=np.int64), values=coo.data, dense_shape=coo.shape)

    def _create_gcn_embed(self) -> list:
        """"""
        # adjacency matrix for information propagation
        A = self.A_in

        # generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        # previous embedding (before update): shape = [n_entity + 1, inter_dim]
        pre_embedding = self.weights['entity_embedding']
        g_embeddings = [pre_embedding]

        for k in range(self.n_layer):
            # propagation
            # neighbor_embedding shape = [n_entity + 1, inter_dim]
            if self.n_fold == 1:
                neighbor_embedding = tf.sparse_tensor_dense_matmul(
                    A_fold_hat[0], pre_embedding, name='gcn_neighbor_%d' % k)
            else:
                temp_embed = []
                for i in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(
                        A_fold_hat[i], pre_embedding))
                neighbor_embedding = tf.concat(temp_embed, axis=0, name='gcn_neighbor_%d' % k)

            # aggregation
            # (W(eh + eNh) + b)
            pre_embedding = neighbor_embedding + pre_embedding
            pre_embedding = tf.nn.leaky_relu(
                tf.matmul(pre_embedding, self.weights['w_gcn_%d' % k]) + self.weights['b_gcn_%d' % k],
                name='gcn_agg_%d' % k)

            # dropout 
            pre_embedding = tf.nn.dropout(
                pre_embedding, rate=self.mess_dropout[k], name='gcn_dropout_%d' % k)
            
            # normalize the distribution of entity embeddings
            norm_embeddings = tf.math.l2_normalize(pre_embedding, axis=1, name='gcn_norm_%d' % k)

            # concatenate information from different layers
            g_embeddings += [norm_embeddings]

        g_embeddings = tf.concat(g_embeddings, 1)

        return g_embeddings

    def _create_lightgcn_embed(self) -> list:
        """"""
        # adjacency matrix for information propagation
        A = self.A_in

        # generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        # previous embedding (before update): shape = [n_entity + 1, inter_dim]
        pre_embedding = self.weights['entity_embedding']
        g_embeddings = [pre_embedding]

        for k in range(self.n_layer):
            # propagation
            # neighbor_embedding shape = [n_entity + 1, inter_dim]
            if self.n_fold == 1:
                neighbor_embedding = tf.sparse_tensor_dense_matmul(
                    A_fold_hat[0], pre_embedding, name='lightgcn_neighbor_%d' % k)
            else:
                temp_embed = []
                for i in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(
                        A_fold_hat[i], pre_embedding))
                neighbor_embedding = tf.concat(temp_embed, axis=0, name='lightgcn_neighbor_%d' % k)

            # aggregation
            # (W(eh + eNh) + b)
            pre_embedding = neighbor_embedding + pre_embedding

            # dropout 
            pre_embedding = tf.nn.dropout(
                pre_embedding, rate=self.mess_dropout[k], name='lightgcn_dropout_%d' % k)
            
            # normalize the distribution of entity embeddings
            norm_embeddings = tf.math.l2_normalize(pre_embedding, axis=1, name='lightgcn_norm_%d' % k)

            # concatenate information from different layers
            g_embeddings += [norm_embeddings]

        g_embeddings = tf.concat(g_embeddings, 1)

        return g_embeddings

    def _create_gnn_embed(self) -> list:
        """"""
        # adjacency matrix for information propagation
        A = self.A_in

        # generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        # previous embedding (before update): shape = [n_entity + 1, inter_dim]
        pre_embedding = self.weights['entity_embedding']
        g_embeddings = [pre_embedding]

        for k in range(self.n_layer):
            # propagation
            # neighbor_embedding shape = [n_entity + 1, inter_dim]
            if self.n_fold == 1:
                neighbor_embedding = tf.sparse_tensor_dense_matmul(
                    A_fold_hat[0], pre_embedding, name='gnn_neighbor_%d' % k)
            else:
                temp_embed = []
                for i in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(
                        A_fold_hat[i], pre_embedding))
                neighbor_embedding = tf.concat(temp_embed, axis=0, name='gnn_neighbor_%d' % k)

            # aggregation: (W(eh || eNh) + b)
            pre_embedding = tf.concat([pre_embedding, neighbor_embedding], 1)
            pre_embedding = tf.nn.leaky_relu(
                tf.matmul(pre_embedding, self.weights['w_gnn_%d' % k]) + self.weights['b_gnn_%d' % k],
                name='gnn_agg_%d' % k)

            # dropout 
            pre_embedding = tf.nn.dropout(
                pre_embedding, rate=self.mess_dropout[k], name='gnn_dropout_%d' % k)

            # normalize the distribution of entity embeddings
            norm_embeddings = tf.math.l2_normalize(pre_embedding, axis=1, name='gnn_norm_%d' % k)

            # concatenate information from different layers
            g_embeddings += [norm_embeddings]

        g_embeddings = tf.concat(g_embeddings, 1)
        # g_embeddings = tf.reduce_sum(g_embeddings, axis=0)

        return g_embeddings
        
    def _create_kgat_embed(self) -> list:
        """"""
        # adjacency matrix for information propagation
        A = self.A_in

        # generate a set of adjacency sub-matrix
        A_fold_hat = self._split_A_hat(A)

        # previous embedding (before update): shape = [n_entity + 1, inter_dim]
        pre_embedding = self.weights['entity_embedding']
        g_embeddings = [pre_embedding]

        for k in range(self.n_layer):
            temp_embed = []
            if self.n_fold == 1:
                neighbor_embedding = tf.sparse_tensor_dense_matmul(
                    A_fold_hat[0], pre_embedding, name='kgat_neighbor_%d' % k)
            else:
                for i in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(
                        A_fold_hat[i], pre_embedding))
                neighbor_embedding = tf.concat(temp_embed, axis=0, name='kgat_neighbor_%d' % k)

            # LeakyReLU (W1(eh + eNh) + b)
            add_embedding = neighbor_embedding + pre_embedding
            sum_embedding = tf.nn.leaky_relu(tf.matmul(add_embedding, self.weights['w1_kgat_%d' % k]) + self.weights['b1_kgat_%d' % k])

            # LeakyReLU (W2(eh ⊙ eNh))
            dot_embedding = tf.multiply(neighbor_embedding, pre_embedding)
            bi_embedding = tf.nn.leaky_relu(tf.matmul(dot_embedding, self.weights['w2_kgat_%d' % k]) + self.weights['b2_kgat_%d' % k])

            pre_embedding = sum_embedding + bi_embedding

            # dropout
            pre_embedding = tf.nn.dropout(pre_embedding, rate=self.mess_dropout[k], name='kgat_dropout_%d' % k)

            # normalize the distribution of entity embeddings
            norm_embeddings = tf.math.l2_normalize(pre_embedding, axis=1, name='kgat_norm_%d' % k)

            g_embeddings += [norm_embeddings]
        
        g_embeddings = tf.concat(g_embeddings, 1)

        return g_embeddings

    def _create_ggnn_embed(self) -> list:
        """"""
        # adjacency matrix for information propagation
        A = self.A_in

        # generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(A)

        # previous embedding (before update): shape = [n_entity + 1, inter_dim]
        pre_embedding = self.weights['entity_embedding']
        g_embeddings = [pre_embedding]

        for k in range(self.n_layer):
            # propagation
            # neighbor_embedding shape = [n_entity + 1, inter_dim]
            if self.n_fold == 1:
                neighbor_embedding = tf.sparse_tensor_dense_matmul(
                    A_fold_hat[0], pre_embedding, name='ggnn_neighbor_%d' % k)
            else:
                temp_embed = []
                for i in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(
                        A_fold_hat[i], pre_embedding))
                neighbor_embedding = tf.concat(temp_embed, axis=0, name='ggnn_neighbor_%d' % k)

            # aggregation
            # GRU(eNh, eh)
            pre_embedding = self.weights['ggnn_gru'](neighbor_embedding, pre_embedding)[1]

            # dropout 
            pre_embedding = tf.nn.dropout(
                pre_embedding, rate=self.mess_dropout[k], name='ggnn_dropout_%d' % k)
            
            # normalize the distribution of entity embeddings
            norm_embeddings = tf.math.l2_normalize(pre_embedding, axis=1, name='ggnn_norm_%d' % k)

            # concatenate information from different layers
            g_embeddings += [norm_embeddings]

        g_embeddings = tf.concat(g_embeddings, 1)

        return g_embeddings

    def _build_classification_model(self):
        f_classification_e = tf.nn.embedding_lookup(self.g_embedding, self.f_classification)
        f_classification_e = tf.nn.l2_normalize(f_classification_e, 1)

        # max pooling entity embeddings in a function
        # f_classification shape: [batch_size, inter_dim]
        f_classification_p = tf.layers.average_pooling1d(
            inputs=f_classification_e,
            pool_size=self.max_entity_func, strides=1,
            name='f_classification_p')
        f_classification_p = tf.reshape(f_classification_p, [-1, self.inter_dim])
        f_classification_p = tf.nn.l2_normalize(f_classification_p, 1)
        f_classification_p = tf.nn.dropout(
                f_classification_p, rate=self.mess_dropout[0])

        # fully connected layer: f_classification_fc shape: [batch_size, fc_dim]
        self.f_classification_fc = tf.matmul(f_classification_p, self.weights['w_classification']) + self.weights['b_classification']

        # code classification prediction
        self.classification_prediction = self.f_classification_fc

    def _build_classification_loss(self):
        """"""
        # code classification applies cross entropy as the loss function
        classification_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_classification,
            logits=self.f_classification_fc,
            name='classification_cross_entropy')
        self.classification_loss = tf.reduce_mean(classification_cross_entropy, name='classification_loss')

         # rep optimization
        if self.opt_type in ['Adam', 'adam']:
            self.classification_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.classification_loss)
        elif self.opt_type in ['SGD', 'sgd']:
            self.classification_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.classification_loss)
        elif self.opt_type in ['AdaDelta']:
            self.classification_opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr).minimize(self.classification_loss)
        else:
            log.error('Optimization is unknown')    
            exit(-1)

        log.info('Finish building loss for code classification')

    def _build_cluster_model(self):
        f_cluster_e = tf.nn.embedding_lookup(self.weights['entity_embedding'], self.f_cluster)
        f_cluster_e = tf.nn.l2_normalize(f_cluster_e, 1)

        # max pooling entity embeddings in a function
        # f_cluster shape: [batch_size, entity_dim]
        f_cluster_p = tf.layers.average_pooling1d(
            inputs=f_cluster_e,
            pool_size=self.max_entity_func, strides=1,
            name='f_cluster_p')
        f_cluster_p =tf.reshape(f_cluster_p, [-1, self.entity_dim])

        # This norm may not be necessary
        norm_f_cluster_e = tf.nn.l2_normalize(f_cluster_p, 1)

        self.f_cluster_rep = norm_f_cluster_e

    def _build_clone_model(self):
        """"""
        f1_clone_e = tf.nn.embedding_lookup(self.g_embedding, self.f1_clone)
        f2_clone_e = tf.nn.embedding_lookup(self.g_embedding, self.f2_clone)
        f1_clone_e = tf.nn.l2_normalize(f1_clone_e, 1)
        f2_clone_e = tf.nn.l2_normalize(f2_clone_e, 1)

        # max pooling entity embeddings in a function
        # f1_clone shape: [batch_size, inter_dim]
        f1_clone_p = tf.layers.average_pooling1d(
            inputs=f1_clone_e,
            pool_size=self.max_entity_func, strides=1,
            name='f1_clone_p')
        f1_clone_p = tf.reshape(f1_clone_p, [-1, self.inter_dim])

        f2_clone_p = tf.layers.average_pooling1d(
            inputs=f2_clone_e,
            pool_size=self.max_entity_func, strides=1,
            name='f2_clone_p')
        f2_clone_p = tf.reshape(f2_clone_p, [-1, self.inter_dim])
        
        """ unsupervised training """
        norm_f1_clone_e = tf.nn.l2_normalize(f1_clone_p, 1)
        norm_f2_clone_e = tf.nn.l2_normalize(f2_clone_p, 1)
        self.clone_prediction_unsupervised = tf.reduce_sum(tf.multiply(norm_f1_clone_e, norm_f2_clone_e), axis=1)

        """ supervised training """
        clone_distance = tf.abs(norm_f1_clone_e - norm_f2_clone_e)
        self.clone_sim = tf.matmul(clone_distance, self.weights['w_clone']) + self.weights['b_clone']
        self.clone_prediction_supervised = self.clone_sim

    def _build_clone_loss(self):
        """"""
        clone_sim = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.y_clone,
            logits=self.clone_sim,
            name='clone_cross_entropy')

        self.clone_loss = tf.reduce_mean(clone_sim, name='clone_loss')

        # rep optimization
        if self.opt_type in ['Adam', 'adam']:
            self.clone_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.clone_loss)
        elif self.opt_type in ['SGD', 'sgd']:
            self.clone_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.clone_loss)
        elif self.opt_type in ['AdaDelta']:
            self.clone_opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr).minimize(self.clone_loss)
        else:
            log.error('Optimization is unknown')    
            exit(-1)

        log.info('Finish building loss for code clone')

    def _statistics_params(self) -> None:
        """"""
        total_parameters = 0
        for var in self.weights:
            if var.find('ggnn') == -1:
                shape = self.weights[var].get_shape()
                var_para = 1
                for dim in shape:
                    var_para *= dim.value
                log.debug('Variable name: %s Shape: %d' % (var, var_para))
                total_parameters += var_para
            else:
                log.debug('Variable name: %s Shape: N/A' % (var))
        log.debug('%s has %d parameters' % (self.model_type, total_parameters))

    def train_classification(self, sess:tf.Session, feed_dict:dict) -> Tuple[float, float]:
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        return sess.run([self.classification_opt, self.classification_loss], feed_dict, options=run_options)

    def train_clone(self, sess:tf.Session, feed_dict:dict) -> Tuple[float, float]:
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        return sess.run([self.clone_opt, self.clone_loss], feed_dict, options=run_options)

    def eval_classification(self, sess:tf.Session, feed_dict:dict) -> list:
        """"""
        return sess.run(self.classification_prediction, feed_dict)

    def eval_clone_unsupervised(self, sess:tf.Session, feed_dict:dict) -> list:
        """"""
        return sess.run(self.clone_prediction_unsupervised, feed_dict)

    def eval_clone_supervised(self, sess:tf.Session, feed_dict:dict) -> list:
        """"""
        return sess.run(self.clone_prediction_supervised, feed_dict)

    def eval_cluster(self, sess:tf.Session, feed_dict:dict) -> list:
        """"""
        return sess.run(self.f_cluster_rep, feed_dict)

    def get_variables_to_restore(self) -> list:
        variables_to_restore = []
        variables = tf.global_variables()
        for v in variables:
            if v.name.split(':')[0] != 'entity_embedding':
                variables_to_restore.append(v)
        return variables_to_restore

    def store_model(self, sess:tf.Session, pretrain_path:str, epoch:int) -> None:
        """"""
        model_save_dir = '%s/%s/%s/%s/%s/' % \
            (pretrain_path, self.model_type, self.lr, self.layer_type, self.reg_type)
        model_save_path = model_save_dir + 'model.weights'
        ensure_dir(model_save_path)

        variables_to_restore = self.get_variables_to_restore()

        model_saver = tf.train.Saver(variables_to_restore, max_to_keep=1)
        model_saver.save(sess, model_save_path, global_step=epoch)

        log.info("Model save in %s" % model_save_path)

    def load_model(self, sess:tf.Session, pretrain_path:str) -> None:
        checkpoint_path = '%s/%s/%s/%s/%s/checkpoint' % \
            (pretrain_path, self.model_type, self.lr, self.layer_type, self.reg_type)
        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_path))
        if ckpt and ckpt.all_model_checkpoint_paths:
            log.info("Load model from %s" % os.path.dirname(checkpoint_path))
            variables_to_restore = self.get_variables_to_restore()
            model_saver = tf.train.Saver(variables_to_restore)
            model_saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
        else:
            log.error("fail to load model in %s" % checkpoint_path)
            exit(-1)
