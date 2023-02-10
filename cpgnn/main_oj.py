import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
import numpy as np
import tensorflow as tf
import random as rd
from time import time

from util.setting import init_setting, log
from util.Word2vec import Word2vec

from model.load_gnn import GNNLoader
from model.SGL import SGL
from model.eval import clone_val_supervised, clone_test_supervised, classification_validation, classification_test

def main():
    """ Get argument settings """
    seed = 2022
    np.random.seed(seed)
    rd.seed(seed)
    tf.set_random_seed(seed)

    """ Initialize args and dataset """
    args = init_setting()
    log.info("Loading data from %s" % args.dataset)
    data_generator = GNNLoader(args)

    """ Use pre-trained word2vec embeddings to initialize AST nodes (entities) """
    type2vec = Word2vec(args)
    if args.pretrain == 0:
        type2vec.init_embedding(data_generator.e2t_list, data_generator.typetoken_seq)
        """ Save pre-trained word2vec embeddings"""
        if args.save_model:
            type2vec.store_embedding(data_generator.out_path)
    elif args.pretrain == 1:
        type2vec.load_embedding(data_generator.out_path)

    """ Select learning models """
    if args.model_type == 'oaktree':
        log.info("Initing Oaktree model")
        model = SGL(args, data_generator, pretrain_embedding=type2vec.embedding)
    else:
        log.error("The ML model is unknown")
        exit(-1)

    """ Setup tensorflow session """
    log.info("Setup tensorflow session")
    sess = model.setup_sess()
    
    """ Reload model parameters for fine tune """
    if args.pretrain == 2:
        model.load_model(sess, data_generator.out_path)

    """ Training phase """
    log.info("Training %d epochs" % args.epoch)
    best_classification_validation = [0]

    epoch = 0
    for epoch in range(args.epoch):
        model.lr = 0.1

        if args.classification_test:
            if epoch > 50:
                model.lr = 0.01

        if args.clone_test_supervised:
            if epoch > 15:
                model.lr = 0.01

        classification_loss, clone_loss = 0., 0.
        t_train = time()

        """ fine-tune for code classification """
        if args.classification_test:
            rd.shuffle(data_generator.classification_train_data)
            for i_batch in range(data_generator.classification_data_iter):
                batch_data = data_generator.generate_classification_train_batch(i_batch)
                feed_dict = data_generator.generate_classification_train_feed_dict(model, batch_data)
                _, classification_loss_batch = model.train_classification(sess, feed_dict)
                classification_loss += classification_loss_batch
            perf_train_ite = 'Epoch %d [%.1fs]: train[lr=%.5f]=[(classification: %.5f)]' % (epoch + 1, time() - t_train, model.lr, classification_loss)

            if np.isnan(classification_loss) == True:
                log.error('error: loss@ is nan')
                exit(-1)

        """ fine-tune for code clone (supervised) """
        if args.clone_test_supervised:
            rd.shuffle(data_generator.clone_train_data)
            for i_batch in range(data_generator.clone_data_iter):
                batch_data = data_generator.generate_clone_train_batch(i_batch)
                feed_dict = data_generator.generate_clone_train_feed_dict(model, batch_data)
                _, clone_loss_batch = model.train_clone(sess, feed_dict)
                clone_loss += clone_loss_batch
            perf_train_ite = 'Epoch %d [%.1fs]: train[lr=%.5f]=[(clone: %.5f)]' % (epoch + 1, time() - t_train, model.lr, clone_loss)

            if np.isnan(clone_loss) == True:
                log.error('error: loss@ is nan')
                exit(-1)

        log.debug(perf_train_ite)

        if epoch % 1 == 0:
            """ classification test """
            if args.classification_test:
                classification_validation(sess, model, data_generator, best_classification_validation)

            """ clone test supervised """
            if args.clone_test_supervised:
                clone_val_supervised(sess, model, data_generator, args.clone_threshold)    

    """ Testing phase """
    if args.clone_test_supervised:
        clone_test_supervised(sess, model, data_generator, args.clone_threshold)

    if args.classification_test:
        classification_test(sess, model, data_generator)

    """ Save the model parameters """
    if args.save_model:
        model.store_model(sess, data_generator.out_path, epoch)

if __name__ == '__main__':
    main()
