from typing import Tuple
import numpy as np
import tensorflow as tf
from sklearn import metrics

from util.setting import log

from model.load_gnn import GNNLoader
from model.SGL import SGL

def calculate_metrics(tp:int, fn:int, tn:int, fp:int) -> Tuple[float, float, float]:
    """"""
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return recall, precision, f1

def calculate_tpr(tp:int, fn:int) -> float:
    tpr = tp / (tp + fn)
    return tpr

def print_metrics(msg:str, rel:dict) -> None:
    """"""
    if rel['valid']:
        log.info('%s: [tp, fn, tn, fp]==[%d, %d, %d, %d] [rec, pre, f1, auc]==[%f, %f, %f, %f]'
                % (msg, 
                rel['tp'], rel['fn'], rel['tn'], rel['fp'], 
                rel['recall'], rel['precision'], rel['f1'], rel['auc']))

def print_tpr(msg:str, rel:dict) -> None:
    """"""
    if rel['valid']:
        log.info('%s: [tp, fn]==[%d, %d] [tpr]==[%f]'
                % (msg, rel['tp'], rel['fn'], rel['tpr']))

def classification_validation(sess:tf.Session, model:SGL, data_generator:GNNLoader, best_classification_validation:list) -> dict:
    """"""
    rel = {'valid': 1, 'recall': 0., 'precision': 0., 'f1': 0., 'accuracy':0.}

    if data_generator.n_classification_val == 0:
        rel['valid'] = 0
        log.warn("n_classification_val == 0")
        return rel

    n_batch_classification_val = data_generator.n_classification_val // data_generator.batch_size_classification
    if n_batch_classification_val == 0:
        n_batch_classification_val = 1
    elif data_generator.n_classification_val % data_generator.batch_size_classification:
        n_batch_classification_val += 1
    
    classification_rel = []
    classification_label = []

    for i_batch in range(n_batch_classification_val):
        batch_data = data_generator.generate_classification_val_batch(i_batch, (i_batch == n_batch_classification_val - 1))
        feed_dict = data_generator.generate_classification_val_feed_dict(model, batch_data)
        
        classification_rel.extend(model.eval_classification(sess, feed_dict))
        classification_label.extend(batch_data['y_classification'])

    classification_pred = np.argmax(classification_rel, axis=1)    
    rel['accuracy'] = metrics.accuracy_score(classification_label, classification_pred)

    if rel['accuracy'] > best_classification_validation[0]:
        best_classification_validation[0] = rel['accuracy']
    
    log.info('Classification Validation: [acc, best]==[%f, %f]' % (rel['accuracy'], best_classification_validation[0]))

def classification_test(sess:tf.Session, model:SGL, data_generator:GNNLoader) -> None:
    """"""
    rel = {'valid': 1, 'recall': 0., 'precision': 0., 'f1': 0., 'accuracy': 0.}

    if data_generator.n_classification_test == 0:
        rel['valid'] = 0
        log.warn("n_classification_test == 0")
        return rel

    n_batch_classification_test = data_generator.n_classification_test // data_generator.batch_size_classification
    if n_batch_classification_test == 0:
        n_batch_classification_test = 1
    elif data_generator.n_classification_test % data_generator.batch_size_classification:
        n_batch_classification_test += 1
    
    classification_rel = []
    classification_label = []

    for i_batch in range(n_batch_classification_test):
        batch_data = data_generator.generate_classification_test_batch(i_batch, (i_batch == n_batch_classification_test - 1))
        feed_dict = data_generator.generate_classification_val_feed_dict(model, batch_data)
        
        classification_rel.extend(model.eval_classification(sess, feed_dict))
        classification_label.extend(batch_data['y_classification'])

    classification_pred = np.argmax(classification_rel, axis=1)    
    rel['precision'], rel['recall'], rel['f1'], _ = metrics.precision_recall_fscore_support(classification_label, classification_pred, average='macro')
    rel['accuracy'] = metrics.accuracy_score(classification_label, classification_pred)

    log.info('Classification Test: [rec, pre, f1, acc]==[%f, %f, %f, %f]'
        % (rel['recall'], rel['precision'], rel['f1'], rel['accuracy']))

def clone_val_supervised(sess:tf.Session, model:SGL, data_generator:GNNLoader, threshold: float) -> None:
    rel = {'valid': 1, 'recall': 0., 'precision': 0., 'f1': 0., 'tpr': 0., 'auc': 0.}

    n_batch_clone = data_generator.n_clone_val // data_generator.batch_size_clone
    if data_generator.n_clone_val % data_generator.batch_size_clone:
        n_batch_clone += 1
    
    clone_rel = []
    clone_label = []

    for i_batch in range(n_batch_clone):
        batch_data = data_generator.generate_clone_val_batch(i_batch, (i_batch == n_batch_clone - 1))
        feed_dict = data_generator.generate_clone_feed_dict(model, batch_data)

        clone_rel.extend(model.eval_clone_supervised(sess, feed_dict))
        clone_label.extend(batch_data['y_clone'])

    clone_pred = np.array(clone_rel) > threshold

    rel['recall'] = metrics.recall_score(clone_label, clone_pred, average='binary')
    rel['precision'] = metrics.precision_score(clone_label, clone_pred, average='binary')
    rel['f1'] = metrics.f1_score(clone_label, clone_pred, average='binary')

    # note: input clone_rel rather than clone_pred
    fpr, tpr, thresholds = metrics.roc_curve(clone_label, clone_rel, pos_label=1)
    rel['auc'] = metrics.auc(fpr, tpr)

    log.info('Clone Validation: [rec, pre, f1, auc]==[%f, %f, %f, %f]'
        % (rel['recall'], rel['precision'], rel['f1'], rel['auc']))

def clone_test_supervised(sess:tf.Session, model:SGL, data_generator:GNNLoader, threshold: float) -> None:
    rel = {'valid': 1, 'recall': 0., 'precision': 0., 'f1': 0., 'tpr': 0., 'auc': 0.}

    n_batch_clone = data_generator.n_clone_test_supervised // data_generator.batch_size_clone
    if data_generator.n_clone_test_supervised % data_generator.batch_size_clone:
        n_batch_clone += 1
    
    clone_rel = []
    clone_label = []
    for i_batch in range(n_batch_clone):
        batch_data = data_generator.generate_clone_test_batch(i_batch, (i_batch == n_batch_clone - 1))
        feed_dict = data_generator.generate_clone_feed_dict(model, batch_data)
        clone_rel.extend(model.eval_clone_supervised(sess, feed_dict))
        clone_label.extend(batch_data['y_clone'])

    clone_pred = np.array(clone_rel) > threshold

    rel['recall'] = metrics.recall_score(clone_label, clone_pred, average='binary')
    rel['precision'] = metrics.precision_score(clone_label, clone_pred, average='binary')
    rel['f1'] = metrics.f1_score(clone_label, clone_pred, average='binary')

    # note: input clone_rel rather than clone_pred
    fpr, tpr, thresholds = metrics.roc_curve(clone_label, clone_rel, pos_label=1)
    rel['auc'] = metrics.auc(fpr, tpr)

    log.info('Clone Test: [rec, pre, f1, auc]==[%f, %f, %f, %f]'
        % (rel['recall'], rel['precision'], rel['f1'], rel['auc']))

def clone_test_unsupervised(sess:tf.Session, model:SGL, data_generator:GNNLoader, threshold:float) -> None:
    clone_threshold = threshold

    rel = {'tp': 0 , 'fn': 0, 'tn': 0, 'fp': 0, 'valid': 1, 
    'recall': 0., 'precision': 0., 'f1': 0., 'tpr': 0., 'auc': 0.}

    n_batch_clone = data_generator.n_clone_test_unsupervised // data_generator.batch_size_clone
    if data_generator.n_clone_test_unsupervised % data_generator.batch_size_clone:
        n_batch_clone += 1
    
    pos_rel = []
    neg_rel = []

    for i_batch in range(n_batch_clone):
        batch_data_pos = data_generator.generate_clone_batch(i_batch, (i_batch == n_batch_clone - 1), pos=True)
        feed_pos = data_generator.generate_clone_feed_dict(model, batch_data_pos)
        
        batch_data_neg = data_generator.generate_clone_batch(i_batch, (i_batch == n_batch_clone - 1), pos=False)
        feed_neg = data_generator.generate_clone_feed_dict(model, batch_data_neg)

        pos_rel.extend(model.eval_clone_unsupervised(sess, feed_pos))
        neg_rel.extend(model.eval_clone_unsupervised(sess, feed_neg))

    pos_pred = np.array(pos_rel) >= clone_threshold
    tp = np.sum(pos_pred)
    fn = pos_pred.shape[0] - tp
    rel['tp'] += tp
    rel['fn'] += fn

    neg_pred = np.array(neg_rel) >= clone_threshold
    fp = np.sum(neg_pred)
    tn = neg_pred.shape[0] - fp
    rel['fp'] += fp
    rel['tn'] += tn

    y = [1] * len(pos_rel) + [-1] * len(neg_rel)
    scores = pos_rel + neg_rel
    fpr, tpr, _ = metrics.roc_curve(y, scores)
    rel['auc'] = metrics.auc(fpr, tpr)

    rel['recall'], rel['precision'], rel['f1'] = calculate_metrics(rel['tp'], rel['fn'], rel['tn'], rel['fp'])
    
    print_metrics("Clone Test Unsupervised", rel)
