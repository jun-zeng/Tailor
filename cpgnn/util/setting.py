import argparse
import logging
from colorlog import ColoredFormatter

log = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(prog="driver",
                                    description="learning code representations")
    # setting for envs
    parser.add_argument('-l', '--logging', type=int, default=10,
                        help='Log level [10-50] (default: 10 - Debug)')
    parser.add_argument('--gpu_id', type=str, default='0,1',
                        help='GPU device id (by default 0,1)')

    # setting for dataset
    parser.add_argument('--dataset', type=str, default='oj',
                        help='Dir to store code encodings (default: oj)')
    parser.add_argument('--splitlabel', default=False, action='store_true',
                        help='Generate labels for dataset splitting')
    parser.add_argument('--cpg_no_cfg', default=False, action='store_true',
                        help='No Control Flow Graph in CPG')
    parser.add_argument('--cpg_no_dfg', default=False, action='store_true',
                        help='No Data Flow Graph in CPG')

    # setting for model
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with word2vec embeddings, 2:Pretrain with stored models. (default: 0)')
    parser.add_argument('--model_type', type=str, default='oaktree',
                        help='type of learning model from {oaktree} (default: oaktree)')
    parser.add_argument('--adj_type', type=str, default='si',
                        help='type of adjacency matrix from {bi, si}. (default: si)')
                        
    # setting for training
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of training epoch')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--regs', nargs='?', default='[1e-4,1e-4]',
                        help='Regularization for entity embeddings.')
    parser.add_argument('--opt_type', type=str, default='Adam',
                        help='type of training optimizer from {Adam, SGD, AdaDelta}')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1,0.1,0.1,0.1]',
                        help='drop probability')
    parser.add_argument('--early_stop', default=True, action='store_false',
                        help='early stop as the performance on validation sets starts to degrade')
                        
    # setting for Word2vec
    parser.add_argument('--type_dim', type=int, default=16,
                        help='embedding size for AST type/token')
    parser.add_argument('--word2vec_window', type=int, default=10,
                        help='context window size for sequences')
    parser.add_argument('--word2vec_count', type=int, default=1,
                        help='frequency are dropped before training occurs')
    parser.add_argument('--word2vec_worker', type=int, default=64,
                        help='number of workers to train word2vec')
    parser.add_argument('--word2vec_save', default=False, action='store_true',
                        help='whether save word2vec embeddings')
    parser.add_argument('--embed_init', type=str, default='comb',
                        help='different embedding initialization from {type, token, comb}')

    # setting for Node2vec
    parser.add_argument('--num_walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')
    parser.add_argument('--walk_length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

    # setting for GNN
    # entity_dim = 2 * type_dim
    # parser.add_argument('--entity_dim', type=int, default=32,
                        # help='embedding size for entities')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,32,32]',
                        help='embedding size for gnn layers (changed with mess_dropout)')
    parser.add_argument('--agg_type', nargs='?', default='gnn',
                        help='type of gnn aggregation from {none, gnn, gcn, lightgcn, ggnn}.')

    parser.add_argument('--save_model', default=False, action='store_true',
                        help='whether save SGL model parameters.')

    # setting for tasks
    parser.add_argument('--classification_num', type=float, default=104,
                        help='the number of classes considered in OJclone datasets')
    # code clone
    parser.add_argument('--clone_threshold', type=float, default=0.5,
                        help='cosine similarity code clone task')
    parser.add_argument('--batch_size_clone', type=int, default=128,
                        help='batch size for code clone detection')

    # unsupervised
    parser.add_argument('--clone_test_unsupervised', default=False, action='store_true',
                        help='whether to test code clone task.')
    # supervised
    parser.add_argument('--clone_test_supervised', default=False, action='store_true',
                        help='whether to test code clone task.')
    parser.add_argument('--clone_val_size', type=float, default=0.1,
                        help='Size of validation dataset for code clone')
    parser.add_argument('--clone_test_size', type=float, default=0.1,
                        help='Size of test dataset for code clone')

    # code classification (supervised)
    parser.add_argument('--classification_test', default=False, action='store_true',
                        help='whether to test code classification task.')     
    parser.add_argument('--class_val_size', type=float, default=0.1,
                        help='Size of validation dataset for code classification')
    parser.add_argument('--class_test_size', type=float, default=0.1,
                        help='Size of test dataset for code classification')
    parser.add_argument('--batch_size_classification', type=int, default=384,
                        help='batch size for code classification')

    # code cluster (unsupervised)
    parser.add_argument('--cluster_test', default=False, action='store_true',
                        help='whether to test code cluster task.') 
    
    # training log report
    parser.add_argument('--report', type=str, default="",
                        help='file name to report training logs.') 
    
    args = parser.parse_args()

    if args.clone_test_unsupervised and args.clone_test_supervised:
        log.error('cannot train with both unsupervised and supervised code clone enabled')
        # exit(-1)

    if args.classification_test == False and args.clone_test_supervised == False:
        log.error('choose training either code representations or code classification or code clone')
        # exit(-1)

    return args
    
def init_logging(level:int, log_file:str) -> None:
    if log_file == "":
        formatter = ColoredFormatter(
            "%(white)s%(asctime)10s | %(log_color)s%(levelname)6s | %(log_color)s%(message)6s",
            reset=True,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'yellow',
                'WARNING':  'green',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            },
        )
        handler = logging.StreamHandler()
    else:
        formatter = logging.Formatter("%(asctime)10s | %(levelname)6s | %(message)6s")
        handler = logging.FileHandler("log/"+log_file+".txt", 'w')

    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(level)

def init_gpu(gpu_id_args:str) -> list:
    gpu_id = list(set(gpu_id_args.split(",")))
    if len(gpu_id) < 2:
        log.error("Please define two GPU cards. You only define %d GPU", len(gpu_id))
        exit(-1)
    else:
        log.info("Use GPU %s and GPU %s for training", gpu_id[0], gpu_id[1])
    return gpu_id

def init_setting() -> argparse.Namespace:
    args = parse_args()
    init_logging(args.logging, args.report)
    args.gpu_id = init_gpu(args.gpu_id)
    return args
