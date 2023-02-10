# Training Log of CPGNN

To demonstrate the **reproducibility of experimental results reported in the paper**
and **facilitate future research**, we provide the training logs of CPGNN on
the OJClone and BigCloneBench datasets for both tasks of code clone detection and
source code classification. The encodings of OJClone and BigCloneBench can be
found at `Tailor/cpgnn/data` in our docker image.

The training logs for the best performances (reported in Table II,III,IV,V in
the paper) are
[classification_oj_layer_5.txt](gnn_layer/classification_oj_layer_5.txt),
[clone_bcb_layer_4.txt](gnn_layer/clone_bcb_layer_4.txt), and
[clone_oj_layer_5.txt](gnn_layer/clone_oj_layer_5.txt).

Please note that we also include training logs for our study of CPGNN as follows:
* `embedding_initialization`: This folder shows the effects of different
  embedding initializations schemas.
* `gnn_variants`: This folder shows the effects of CPGNN variants.
* `gnn_layer`: This folder shows the effects of propagation iteration numbers.
* `cpg_representations`: This folder shows the effects of alternative CPG representations.

# Example
Our training log for code clone detection on the OJClone dataset is shown as
follows:
```batch
2022-02-19 19:08:57,880 |   INFO | Loading data from oj_clone_encoding
2022-02-19 19:08:58,848 |   INFO | Extracting statements
2022-02-19 19:09:00,638 |   INFO | Extracting interactions
2022-02-19 19:09:06,134 |  DEBUG | CPG statistics
2022-02-19 19:09:06,134 |  DEBUG | [n_typetoken, n_entity, n_stat, n_relation] = [3732, 246363, 242272, 8]
2022-02-19 19:09:06,134 |  DEBUG | [n_triple, n_inter] = [1005433, 1148858]
2022-02-19 19:09:06,134 |  DEBUG | [n_ast, n_cfg, n_pdg] = [593110, 339805, 215943]
2022-02-19 19:09:06,134 |  DEBUG | [max n_entity for a statement] = [161]
2022-02-19 19:09:06,134 |   INFO | Parsing code clone/classification dataset
2022-02-19 19:09:06,135 | WARNING | Extract only the first 15 functionalities from ojclone dataset
2022-02-19 19:09:06,135 |  DEBUG | The total number of clone functions: 7500
2022-02-19 19:09:06,626 |   INFO | Converting interactions into sparse adjacency matrix
2022-02-19 19:09:06,795 |   INFO | Generating normalized sparse adjacency matrix
2022-02-19 19:09:07,003 |   INFO | Generating code clone training, validation, and testing sets
2022-02-19 19:09:17,528 |  DEBUG | Code Clone [n_train, n_val, n_test] = [255840, 31980, 31980]
2022-02-19 19:09:17,528 |   INFO | Initing type/token embeddings with word2vec
2022-02-19 19:09:24,584 |   INFO | Initing Oaktree model
2022-02-19 19:09:24,590 |   INFO | Finish building inputs for SGL
2022-02-19 19:09:25,029 |   INFO | Init entity embeddings with pre-trained word2vec embeddings
2022-02-19 19:09:25,095 |   INFO | Finish building weights for SGL
2022-02-19 19:09:25,420 |   INFO | Finish building model for GNN
2022-02-19 19:09:26,592 |   INFO | Finish building loss for code clone
2022-02-19 19:09:26,593 |  DEBUG | Variable name: entity_embedding Shape: 7883648
2022-02-19 19:09:26,593 |  DEBUG | Variable name: w_gnn_0 Shape: 2048
2022-02-19 19:09:26,593 |  DEBUG | Variable name: b_gnn_0 Shape: 32
2022-02-19 19:09:26,593 |  DEBUG | Variable name: w_gnn_1 Shape: 2048
2022-02-19 19:09:26,593 |  DEBUG | Variable name: b_gnn_1 Shape: 32
2022-02-19 19:09:26,593 |  DEBUG | Variable name: w_gnn_2 Shape: 2048
2022-02-19 19:09:26,593 |  DEBUG | Variable name: b_gnn_2 Shape: 32
2022-02-19 19:09:26,593 |  DEBUG | Variable name: w_gnn_3 Shape: 2048
2022-02-19 19:09:26,593 |  DEBUG | Variable name: b_gnn_3 Shape: 32
2022-02-19 19:09:26,593 |  DEBUG | Variable name: w_gnn_4 Shape: 2048
2022-02-19 19:09:26,593 |  DEBUG | Variable name: b_gnn_4 Shape: 32
2022-02-19 19:09:26,593 |  DEBUG | Variable name: w_clone Shape: 192
2022-02-19 19:09:26,593 |  DEBUG | Variable name: b_clone Shape: 1
2022-02-19 19:09:26,593 |  DEBUG | oaktree_si_gnn has 7894241 parameters
2022-02-19 19:09:26,593 |   INFO | Setup tensorflow session
2022-02-19 19:09:27,883 |   INFO | Training 30 epochs
2022-02-19 19:12:48,993 |  DEBUG | Epoch 1 [201.1s]: train[lr=0.10000]=[(clone: 77.46563)]
2022-02-19 19:13:05,593 |   INFO | Clone Validation: [rec, pre, f1, auc]==[0.000000, 0.000000, 0.000000, 0.987570]
2022-02-19 19:16:18,984 |  DEBUG | Epoch 2 [193.4s]: train[lr=0.10000]=[(clone: 34.83083)]
2022-02-19 19:16:35,326 |   INFO | Clone Validation: [rec, pre, f1, auc]==[0.716667, 0.955556, 0.819048, 0.991236]
...
2022-02-19 20:50:28,935 |  DEBUG | Epoch 29 [192.9s]: train[lr=0.01000]=[(clone: 0.77707)]
2022-02-19 20:50:45,548 |   INFO | Clone Validation: [rec, pre, f1, auc]==[0.996465, 1.000000, 0.998229, 0.999999]
2022-02-19 20:53:58,175 |  DEBUG | Epoch 30 [192.6s]: train[lr=0.01000]=[(clone: 0.23005)]
2022-02-19 20:54:14,423 |   INFO | Clone Validation: [rec, pre, f1, auc]==[0.996970, 0.999494, 0.998230, 1.000000]
2022-02-19 20:54:30,588 |   INFO | Clone Test: [rec, pre, f1, auc]==[0.997475, 1.000000, 0.998736, 1.000000]
```

Where:
* `Epoch 30 [192.6s]:` shows the time cost for one epoch training.
* `[(clone: 0.23005)]` shows the training loss for code clone detection.
* `[rec, pre, f1, auc]` shows the performance in terms of recall, precision,
  f-score, and auc.
* `Clone Validation` and `Clone Test` shows the results of the validation and
  testing sets, respectively.

# Clarification
Here we would like to clarify some points:
* The time cost will be different based on the running machines for CPGNN
  training.
* The training loss will be slightly different based on the random seeds.
* We have cleaned up sources files (e.g., SGL_constractive.py to
  SGL.py) for artifact evaluation. You may find name inconsistencies between the source files and
  the logs.