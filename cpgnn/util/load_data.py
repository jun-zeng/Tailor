import os
import argparse
import numpy as np
import random as rd
from typing import Tuple

from util.setting import log
from util.helper import ensure_dir, exist_dir
from random_choice import randint_choice

class Data(object):
    def __init__(self, args:argparse.Namespace) -> None:
        super().__init__()
        # generate labels for data splitting
        self.split_label = args.splitlabel

        # parse code clone/classification data
        self.classification_num = int(args.classification_num)

        # batch_size for code clone (both supervised and unsupervised)
        self.batch_size_clone = args.batch_size_clone

        # batch_size for code cluster
        self.batch_size_cluster = 512

        # batch_size for code classification
        self.batch_size_classification = args.batch_size_classification

        # # encoding and embedding file paths
        self.in_path, self.out_path = self.init_path(args.dataset)

        # # init file_path to load cpg
        self.entity_file = os.path.join(self.in_path, 'entity2id.txt')
        exist_dir(self.entity_file)
        self.rel_file = os.path.join(self.in_path, 'rel2id.txt')
        exist_dir(self.rel_file)
        self.triple_file = os.path.join(self.in_path, 'triple2id.txt')
        exist_dir(self.triple_file)
        self.stat2entity_file = os.path.join(self.in_path, 'stat2entity.txt')
        exist_dir(self.stat2entity_file)
        self.typetoken2id_file = os.path.join(self.in_path, 'typetoken2id.txt')
        exist_dir(self.typetoken2id_file)
        self.typetoken_seq_file = os.path.join(self.in_path, 'typetoken_seq.txt')
        exist_dir(self.typetoken_seq_file)
        self.entity2typetoken_file = os.path.join(self.in_path, 'entity2typetoken.txt')
        exist_dir(self.entity2typetoken_file)

        # used for graph2vec 
        self.graph2vec_file = os.path.join(self.in_path, 'graph2vec')
        exist_dir(self.graph2vec_file)

        # extract entity to AST type/token information
        self.e2t_list = self._extract_entity2typetoken()
        self.typetoken_seq = self._extract_typetoken_seq()

        # collect statistic info about the dataset
        self.n_typetoken, self.n_entity, self.n_relation, self.n_triple = self._load_cpg_stat()

        # extract entities of program statements
        log.info('Extracting statements')
        self.n_stat, self.max_entity_stat, self.stat_dict = self._extract_stat()

        # extract interactions (do not differentiate ast/cfg/pdg) from cpg
        log.info('Extracting interactions')
        self.cpg_no_cfg = args.cpg_no_cfg
        self.cpg_no_dfg = args.cpg_no_dfg
        self.inter_data, self.inter_dict = self._extract_inter()

        self._print_cpg_info()
        
        self.max_entity_func = 0
        if args.clone_test_unsupervised or args.classification_test or args.cluster_test or args.clone_test_supervised:
            log.info('Parsing code clone/classification dataset')
            if args.dataset.find('bcb') > -1:
                # load bcb dataset
                self.func_file = os.path.join(self.in_path, 'bcb_clone.txt')
                self.codelabel_file = os.path.join(self.in_path, 'clone_labels.txt')
                exist_dir(self.codelabel_file)
                self.max_entity_func, self.all_clone_type = self._extract_bcb_code_clone()
            else:
                # load ojclone dataset  
                self.codeclone_file = os.path.join(self.in_path, 'code_clone.txt')
                exist_dir(self.codeclone_file)
                self.max_entity_func, self.all_clone_family = self._extract_oj_code_clone()

    def _extract_entity2typetoken(self) -> list:
        """"""
        e2t_list = []

        with open(self.entity2typetoken_file, 'r') as f:
            next(f)
            for line in f.readlines():
                e2t = line.strip().split(',')
                type = e2t[1]
                token = e2t[2]
                e2t_list.append([int(type), int(token)])
        
        return e2t_list
        
    def _extract_typetoken_seq(self) -> list:
        """"""
        typetoken_seq = []

        with open(self.typetoken_seq_file, 'r') as f:
            next(f)
            for line in f.readlines():
                seq = list(map(int, line.strip().split(',')))
                typetoken_seq.append(seq)

        return typetoken_seq

    def init_path(self, dataset:str) -> Tuple[str, str]:
        """ """
        # in_path defines where to load code encodings
        in_path = os.path.abspath(os.path.join('data', dataset))
        exist_dir(in_path)

        # output_path defines where to save code embeddings (representations)
        out_path = os.path.abspath(os.path.join('pretrain', dataset))
        ensure_dir(out_path)

        return in_path, out_path

    def _load_cpg_stat(self) -> Tuple[int, int, int]:
        """ """
        with open(self.typetoken2id_file, 'r') as f:
            n_typetoken = int(f.readline().strip())
        with open(self.entity_file, 'r') as f:
            n_entity = int(f.readline().strip())
        with open(self.rel_file, 'r') as f:
            n_relation = int(f.readline().strip())
        with open(self.triple_file, 'r') as f:
            n_triple = int(f.readline().strip())
        return n_typetoken, n_entity, n_relation, n_triple

    def _print_cpg_info(self):
        """ """
        log.debug('CPG statistics')
        log.debug('[n_typetoken, n_entity, n_stat, n_relation] = [%d, %d, %d, %d]'
        % (self.n_typetoken, self.n_entity, self.n_stat, self.n_relation))
        log.debug('[n_triple, n_inter] = [%d, %d]' % (self.n_triple, self.n_inter))
        log.debug('[n_ast, n_cfg, n_pdg] = [%d, %d, %d]' % (self.n_ast, self.n_cfg, self.n_pdg))
        log.debug('[max n_entity for a statement] = [%d]' % self.max_entity_stat)

    def _extract_inter(self) -> Tuple[list, dict]:
        """ """
        inter_mat = list()
        inter_dict = dict()

        # we treat AST as [0]; CFG as [1]: PDG as [2]
        inter_dict[0] = []
        inter_dict[1] = []
        inter_dict[2] = []

        with open(self.triple_file, 'r') as f:
            next(f)
            for line in f.readlines():
                triple = line.strip().split(',')
                h_id = int(triple[0])
                t_id = int(triple[1])
                r_id = int(triple[2])

                # We hardcode AST, CFG, and PDG as 100, 010, and 001
                r_b = "{0:03b}".format(r_id)

                if not (r_b == '100' or r_b == '110' or r_b == '111' or r_b == '010' or r_b == '011' or r_b == '001' or r_b == '101'):
                    log.error('unknown r_id in {}'.format(triple))
                    exit(-1)

                if r_b[0] == '1':
                    inter_dict[0].append([h_id, t_id])
                    inter_mat.append([h_id, t_id])
                if r_b[1] == '1' and self.cpg_no_cfg == False:
                    inter_dict[1].append([h_id, t_id])
                    inter_mat.append([h_id, t_id])
                if r_b[2] == '1' and self.cpg_no_dfg == False:
                    inter_dict[2].append([h_id, t_id])
                    inter_mat.append([h_id, t_id])

        self.n_ast = len(inter_dict[0])
        self.n_cfg = len(inter_dict[1])
        self.n_pdg = len(inter_dict[2])

        inter_data = np.array(inter_mat)
        self.n_inter = len(inter_data)

        return inter_data, inter_dict

    def _extract_stat(self) -> Tuple[int, int, dict]:
        """ """
        stat_dict = dict()
        max_entity_stat = 0

        with open(self.stat2entity_file) as f:
            n_stat = int(f.readline().strip())
            for line in f.readlines():
                entities = [int(i) for i in line.strip().split(',')]
                # entities have repeated entities, e.g., entities = [0,0,41,24]
                if len(entities) > max_entity_stat + 1:
                    max_entity_stat = len(entities) - 1
                stat_dict[entities[0]] = [e for e in list(set(entities))]
            
            for stat in stat_dict:
                if len(stat_dict[stat]) < max_entity_stat:
                    # self.n_entity is a `ghost` entity
                    zero_padding = [self.n_entity] * (max_entity_stat - len(stat_dict[stat]))
                    stat_dict[stat].extend(zero_padding)

        return n_stat, max_entity_stat, stat_dict

    def _extract_bcb_code_clone(self):
        """ """
        # encode code clone data from BigCloneBench dataset
        log.info('Extract functions from bcb dataset')
        func_dict = {}

        # max number of entities for an individual function
        max_entity_func = 0
        
        # load functions encoded by entities
        with open(self.func_file) as f:
            n_func = int(f.readline().strip())
            log.debug('The total number of clone functions: %d' % n_func)

            for line in f.readlines():
                tokens = line.split(',')
                func_id = tokens[0]
                entities = list(map(int, tokens[1:]))
                if self.split_label:
                    entities = [int(func_id)] + entities
                func_dict[func_id] = entities
                if len(entities) > max_entity_func:
                    max_entity_func = len(entities)

            for fun_id, func in func_dict.items():
                if len(func) < max_entity_func:
                    # self.n_entity is a `ghost` entity
                    zero_padding = [self.n_entity] * (max_entity_func - len(func))
                    func.extend(zero_padding)
    
        # 0: Type I, 1: Type II, 2: Strong Type III, 3: Medium Type III, 4: Type
        # IV, 5: True Negative
        all_clone_type = [[] for _ in range(6)]

        # load ground-truth code clones
        with open(self.codelabel_file) as f:
            n_clone_pair = int(f.readline().strip())  

            for line in f.readlines():
                tokens = line.split(',')
                func_1 = func_dict[tokens[0]]
                func_2 = func_dict[tokens[1]]
                type = int(tokens[-1]) # int(tokens[3]) 
            
                all_clone_type[type].append([func_1, func_2])

            assert(n_clone_pair == sum(len(clone_type) for clone_type in all_clone_type))

            log.debug('Code clone: [n_I, n_II, n_SIII, n_MIII, n_VI, n_TN] = [%d, %d, %d, %d, %d, %d]'%(len(all_clone_type[0]), len(all_clone_type[1]), len(all_clone_type[2]), len(all_clone_type[3]), len(all_clone_type[4]), len(all_clone_type[5])))
            
        return max_entity_func, all_clone_type

    def _extract_oj_code_clone(self):
        """ """
        # encode code clone data from ojclone dataset
        log.warn('Extract only the first {} functionalities from ojclone dataset'.format(self.classification_num))
        all_clone_family = [[] for _ in range(self.classification_num)]

        # max number of entities for an individual function
        max_entity_func = 0

        # ojclone file sequence
        file_id = 0

        with open(self.codeclone_file) as f:
            n_func = int(f.readline().strip())
            log.debug('The total number of clone functions: %d' % n_func)

            for line in f.readlines():
                tokens = line.split(',')
                family_num = int(tokens[0]) - 1
                if family_num >= self.classification_num:
                    continue
                entities = list(map(int, tokens[1:]))
                if self.split_label:
                    entities = [file_id] + entities
                    file_id += 1
                all_clone_family[family_num].append(entities)   
                # entities do not have repeated entities
                if len(entities) > max_entity_func:
                    max_entity_func = len(entities)

            for family in all_clone_family:
                for func in family:
                    if len(func) < max_entity_func:
                        # self.n_entity is a `ghost` entity
                        zero_padding = [self.n_entity] * (max_entity_func - len(func))
                        func.extend(zero_padding)

        return max_entity_func, all_clone_family
