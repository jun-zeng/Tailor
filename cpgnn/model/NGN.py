import argparse
from typing import Tuple
import numpy as np
import tensorflow as tf
from sklearn import metrics
from util.helper import log

# no graph neural network
class NGN(object):
    def __init__(self, args:argparse.Namespace, pretrain_embedding:np.array, curr_type=-1) -> None:
        super().__init__()
        """ Parse arguments for SGL """
        self._parse_args(args, pretrain_embedding)

        """ Generate training/validation/test dataset """
        if self.clone_train:
            if args.dataset.find("oj") > -1:
                self._sample_oj_clone_split()
            else:
                self.curr_type = curr_type
                self._sample_bcb_clone_split()

        if self.classification_train:
            self._sample_oj_classification_split()

        """ Create placeholder for training inputs """
        self._build_inputs()

        """ Create variable for training weights """
        self._build_weights()

        """ Build model/loss for code classification """
        if self.classification_train:
            self._build_classification_model()
            self._build_classification_loss()

        """ Build model for code clone """
        if self.clone_train:
            self._build_clone_model()
            self._build_clone_loss()

        """ Count the number of model parameters """
        self._statistics_params()

    def setup_sess(self) -> tf.Session:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # tf_config.log_device_placement = True
        sess = tf.Session(config=tf_config)
        sess.run(tf.global_variables_initializer())
        return sess

    def _parse_args(self, args:argparse.Namespace, pretrain_embedding:np.array) -> None:
        """"""
        self.func_dim = pretrain_embedding.shape[1]
        self.lr = args.lr

        # pretrain word2vec embedding
        self.pretrain_embedding = pretrain_embedding

        # learning flags
        self.clone_train = args.clone_test_supervised
        self.classification_train = args.classification_test

        # setting for code classification
        self.classification_num = int(args.classification_num)

        # split labels used in SGL
        self.oj_clone_label_file = 'pretrain/code2vec/oj_clone_split.txt'
        # self.oj_clone_label_file = 'data/oj_encoding/oj_clone_split_30000.txt'
        self.oj_classification_label_file = 'pretrain/code2vec/oj_classification_split.txt'

        # self.bcb_clone_label_file = 'pretrain/code2vec/bcb_clone_split.txt'
        self.bcb_clone_label_file = 'pretrain/rae/bcb_clone_split_rae.txt'

    def _sample_oj_classification_split(self):
        f = open(self.oj_classification_label_file, 'r')
        
        self.classification_train_f = []
        self.classification_train_label = []

        self.classification_val_f = []
        self.classification_val_label = []
        
        self.classification_test_f = []
        self.classification_test_label = []

        for line in f.readlines():
            tokens = line.split()
            func = int(tokens[0])
            label = int(tokens[1])
            split = int(tokens[2])

            if split == 0:
                self.classification_train_f.append([func])
                self.classification_train_label.extend([label])
            elif split == 1:
                self.classification_val_f.append([func])
                self.classification_val_label.append([label])
            elif split == 2:
                self.classification_test_f.append([func])
                self.classification_test_label.append([label])

        f.close()

        log.debug('Code Classification [n_train, n_val, n_test] = [%d, %d, %d]' % (len(self.classification_train_f), len(self.classification_val_f), len(self.classification_test_f)))

    def _sample_oj_clone_split(self):
        f = open(self.oj_clone_label_file, 'r')

        self.clone_train_f1, self.clone_train_f2 = [], []
        self.clone_val_f1, self.clone_val_f2 = [], []
        self.clone_test_f1, self.clone_test_f2 = [], []

        self.clone_train_label = []
        self.clone_val_label = []
        self.clone_test_label = []

        for line in f.readlines():
            tokens = line.split()
            f1 = int(tokens[0])
            f2 = int(tokens[1])
            label = int(tokens[2])
            split = int(tokens[3])

            if split == 0:
                self.clone_train_f1.append([f1])
                self.clone_train_f2.append([f2])
                self.clone_train_label.append([label])
            elif split == 1:
                self.clone_val_f1.append([f1])
                self.clone_val_f2.append([f2])
                self.clone_val_label.append([label])
            elif split == 2:
                self.clone_test_f1.append([f1])
                self.clone_test_f2.append([f2])
                self.clone_test_label.append([label])

        f.close()

        log.debug('Code Clone [n_train, n_val, n_test] = [%d, %d, %d]' % (len(self.clone_train_f1), len(self.clone_val_f1), len(self.clone_test_f1)))

    def _sample_bcb_clone_split(self):
        f = open(self.bcb_clone_label_file, 'r')

        # 0: Type I, 1: Type II, 2: Strong Type III, 3: Medium Type III, 4: Type
        # IV, 5: True Negative
        self.clone_train_f1, self.clone_train_f2 = [], []
        self.clone_val_f1, self.clone_val_f2 = [], []
        self.clone_test_f1, self.clone_test_f2 = [], []

        self.clone_train_label = []
        self.clone_val_label = []
        self.clone_test_label = []

        for line in f.readlines():
            tokens = line.split()

            type = int(tokens[2])
            if type != self.curr_type and type != 5:
                continue
            
            f1 = int(tokens[0])
            f2 = int(tokens[1])
            split = int(tokens[3])
            if split == 0:
                self.clone_train_f1.append([f1])
                self.clone_train_f2.append([f2])
                if type == 5:
                    self.clone_train_label.append([0])
                else:
                    self.clone_train_label.append([1])
            elif split == 1:
                self.clone_val_f1.append([f1])
                self.clone_val_f2.append([f2])
                if type == 5:
                    self.clone_val_label.append([0])
                else:
                    self.clone_val_label.append([1])
            elif split == 2:
                self.clone_test_f1.append([f1])
                self.clone_test_f2.append([f2])
                if type == 5:
                    self.clone_test_label.append([0])
                else:
                    self.clone_test_label.append([1])

        f.close()
        
        log.debug('Code Clone [n_train, n_val, n_test] = [%d, %d, %d]' % (len(self.clone_train_f1), len(self.clone_val_f1), len(self.clone_test_f1)))
        
    def _build_inputs(self) -> None:
        """"""
        # input function/label for code classification
        self.f_classification = tf.placeholder(dtype=tf.int64, name='f_classification',
            shape=[None, 1])
        self.y_classification = tf.placeholder(dtype=tf.int64, name='y_classification',
            shape=[None])

        # input a pair of functions for code clone
        # f_e shape: [batch_size, entity_num]
        self.f1_clone = tf.placeholder(dtype=tf.int64, name='f1_clone',
            shape=[None, 1])
        self.f2_clone = tf.placeholder(dtype=tf.int64, name='f2_clone',
            shape=[None, 1])
        self.y_clone = tf.placeholder(dtype=tf.float32, name='y_clone',
            shape=[None, 1])

        log.info("Finish building inputs for NGN")

    def _build_weights(self) -> None:
        """"""
        all_weight = dict()
        initializer = tf.contrib.layers.xavier_initializer(seed=2022)
    
        # weights for entity embeddings: fine-tune does not alter entity embeddings
        entity_trainable = False
        if self.pretrain_embedding is None:
            log.error("Do not declare pre-trained code2vec/graph2vec embeddings")
            exit(-1)
        else:
            all_weight['func_embedding'] = tf.Variable(
            initial_value=self.pretrain_embedding,
            trainable=entity_trainable,
            name='func_embedding',dtype=tf.float32)
            log.info("Init function embeddings with pre-trained code2vec/graph2vec embeddings")

        # weights for code classification
        if self.classification_train:
            all_weight['w_classification'] = tf.Variable(
                initial_value=initializer([self.func_dim, self.classification_num]), name='w_classification')
            all_weight['b_classification'] = tf.Variable(
                initial_value=initializer([1, self.classification_num]), name='b_classification')  

        # weights for code clone
        if self.clone_train:
            all_weight['w_clone'] = tf.Variable(
                initial_value=initializer([self.func_dim, 1]), name='w_clone')
            all_weight['b_clone'] = tf.Variable(
                initial_value=initializer([1, 1]), name='b_clone')

        self.weights = all_weight
        
        log.info("Finish building weights for SGL")

    def _build_clone_model(self):
        """"""
        f1_clone_e = tf.nn.embedding_lookup(self.weights['func_embedding'], self.f1_clone)
        f2_clone_e = tf.nn.embedding_lookup(self.weights['func_embedding'], self.f2_clone)
        
        f1_clone_e = tf.reshape(f1_clone_e, [-1, self.func_dim])
        f2_clone_e = tf.reshape(f2_clone_e, [-1, self.func_dim])

        clone_distance = tf.abs(f1_clone_e - f2_clone_e)
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
        self.clone_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.clone_loss)

        log.info('Finish building loss for code clone')

    def _build_classification_model(self):
        f_classification_e = tf.nn.embedding_lookup(self.weights['func_embedding'], self.f_classification)

        f_classification_e = tf.reshape(f_classification_e, [-1, self.func_dim])

        # fully connected layer: f_classification_fc shape: [batch_size, fc_dim]
        self.f_classification_fc = tf.matmul(f_classification_e, self.weights['w_classification']) + self.weights['b_classification']

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
        self.classification_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.classification_loss)

        log.info('Finish building loss for code classification')

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
        log.debug('NGN has %d parameters' % (total_parameters))

    def train_classification(self, sess:tf.Session) -> Tuple[float, float]:
        feed_dict = {
            self.f_classification: self.classification_train_f,
            self.y_classification: self.classification_train_label,
        }
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        return sess.run([self.classification_opt, self.classification_loss], feed_dict, options=run_options)

    def train_clone(self, sess:tf.Session) -> Tuple[float, float]:
        feed_dict = {
            self.f1_clone: self.clone_train_f1,
            self.f2_clone: self.clone_train_f2,
            self.y_clone: self.clone_train_label,
        }
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        return sess.run([self.clone_opt, self.clone_loss], feed_dict, options=run_options)

    def clone_val_supervised(self, sess:tf.Session, threshold:float):
        feed_dict = {
            self.f1_clone: self.clone_val_f1,
            self.f2_clone: self.clone_val_f2,
        }
        clone_rel = sess.run(self.clone_prediction_supervised, feed_dict)
        clone_label = self.clone_val_label

        clone_pred = np.array(clone_rel) > threshold

        recall = metrics.recall_score(clone_label, clone_pred, average='binary')
        precison = metrics.precision_score(clone_label, clone_pred, average='binary')
        f1 = metrics.f1_score(clone_label, clone_pred, average='binary')

        # note: input clone_rel rather than clone_pred
        fpr, tpr, _ = metrics.roc_curve(clone_label, clone_rel, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        log.info('Clone Validation: [rec, pre, f1, auc]==[%f, %f, %f, %f]'
            % (recall, precison, f1, auc))

    def clone_test_supervised(self, sess:tf.Session, threshold:float):
        feed_dict = {
            self.f1_clone: self.clone_test_f1,
            self.f2_clone: self.clone_test_f2,
        }
        clone_rel = sess.run(self.clone_prediction_supervised, feed_dict)
        clone_label = self.clone_test_label

        for threshold in [-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0.0, 0.2,0.3,0.4,0.5]:
            clone_pred = np.array(clone_rel) > threshold

            recall = metrics.recall_score(clone_label, clone_pred, average='binary')
            precison = metrics.precision_score(clone_label, clone_pred, average='binary')
            f1 = metrics.f1_score(clone_label, clone_pred, average='binary')

            # note: input clone_rel rather than clone_pred
            fpr, tpr, _ = metrics.roc_curve(clone_label, clone_rel, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            log.info('Clone Test (threshold %f): [rec, pre, f1, auc]==[%f, %f, %f, %f]'
                % (threshold, recall, precison, f1, auc))

    def classification_validation(self, sess:tf.Session) -> list:
        """"""
        feed_dict = {
            self.f_classification: self.classification_val_f,
        }
        classification_rel = sess.run(self.classification_prediction, feed_dict)
        classification_label = self.classification_val_label

        classification_pred = np.argmax(classification_rel, axis=1)    

        precision, recall, f1, _ = metrics.precision_recall_fscore_support(classification_label, classification_pred, average='macro')
        accuracy = metrics.accuracy_score(classification_label, classification_pred)

        log.info('Classification Validation: [rec, pre, f1, acc]==[%f, %f, %f, %f]'
        % (recall, precision, f1, accuracy))
    
    def classification_test(self, sess:tf.Session) -> list:
        """"""
        feed_dict = {
            self.f_classification: self.classification_test_f,
        }
        classification_rel = sess.run(self.classification_prediction, feed_dict)
        classification_label = self.classification_test_label

        classification_pred = np.argmax(classification_rel, axis=1)    

        precision, recall, f1, _ = metrics.precision_recall_fscore_support(classification_label, classification_pred, average='macro')
        accuracy = metrics.accuracy_score(classification_label, classification_pred)

        log.info('Classification Test: [rec, pre, f1, acc]==[%f, %f, %f, %f]'
        % (recall, precision, f1, accuracy))
    