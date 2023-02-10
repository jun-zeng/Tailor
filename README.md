# Tailor
This repository contains the artifact of our ICSE'23 paper titled
> Learning graph-based code representations for source-level functional similarity detection

# Introduction
Tailor is a new code representation learning framework tailed to detect code functional similarity.
Build upon a customized graph neural network, Tailor explicitly models graph-structured code
features from code property graphs to provide better program functionality classification.

# Overview
* [Getting Started](#getting-started)
    * [Requirement](#requirement)
    * [Install](#install)
* [Status](#status)
* [Repo Structure](#repo-structure)
* [Run Experiments and Validate Results](#run-experiments-and-validate-results)
    * [Usage Guidance](#usage-guidance)
    * [Dataset](#dataset)
    * [Reproducibility](#reproducibility)
* [License](#license)
* [Publication](#publication)
* [Contact](#contact)

# Getting Started
## Requirement
* Hardware requirement:
    GPU: two GPUs, each with 32GB memory; Physical memory: 64 GB
* OS requirement:
    Ubuntu (Our implementation has been tested using Ubuntu 20.04 and 18.04)
* We provide two ways to install Tailor:
    (1) `Build on a docker image provided by us:` You need to install
    [docker](https://docs.docker.com/engine/install/ubuntu/)
    (2) `Build on your machine from scratch:` You need to install
    [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
    and the corresponding CUDA [Version 10.0](https://www.tensorflow.org/install/source#gpu).
    (This implementation has been tested using Python 3.6.5 and tensorflow 1.14)

## Install
We strongly encourage you to use the docker image provided by us. This is the
image we used in our evaluation. We have set up the environment and installed
all the dependencies so you can reproduce the results easily. 

### Build on a docker image provided by us:
* Download Tailor docker image (tailor_image.tar) from Zenodo: https://doi.org/10.5281/zenodo.7533280
* Make sure you have installed docker in Ubuntu
* Load Tailor image: ``$ docker load < tailor_image.tar``
* Initialize a container: ``$ docker run -it --gpus all tailor_image bash`` (If you encounter the
  problem of docker: Error response from daemon: could not select device driver "" with
  capabilities: \[\[gpu\]\]., please install the nvidia-container-toolkit and nvidia-docker2 packages using apt. Note: If you encounter the problem of "Unable to
  locate package nvidia-container-toolkit", please refer to this
  [solution](https://github.com/NVIDIA/nvidia-docker/issues/1238))
* Go to Tailor: ``$ cd /home/Tailor``

### Build on your machine from scratch:
* Download source code and data (Tailor) from https://github.com/jun-zeng/Tailor or Zenodo: https://doi.org/10.5281/zenodo.7533280
* Make sure you have installed Miniconda and CUDA 10.0 in Ubuntu
* Create Tailor virtual environment: `conda env create -f environment.yml`
* Activate Tailor virtual environment: `conda activate tailor`
* Compile Cython: `cd cpgnn && python setup.py build_ext --inplace`

# Status
We apply for the functional and available badges in this artifact evaluation. 
To receive these badges, we have publicly released all the source code and data of Tailor.
Besides, we have documented step-to-step guidance to run experiments and reproduce results reported in our paper. 
We have also prepared a docker image to set up our experimental environment.

# Repo Structure
```
.
├── cpg             # Build code property graph (CPG) from source code
├── cpgnn           # Learn code representations from CPG
├── datasets        # Experimental Dataset
├── log             # Evaluation Logs
└── README.md
```

# Run Experiments and Validate Results
## Usage Guidance
* Code property graph (CPG) construction
    * Show usage and help:
    
    ```bash
    python driver.py -h

    usage: driver [-h] [-l LOGGING] [--src_path SRC_PATH] 
                  [--clone_classification CLONE_CLASSIFICATION] 
                  [--encoding] [--encode_path ENCODE_PATH] 
                  [--iresult_path IRESULT_PATH] [--store_iresult] 
                  [--load_iresult] [--statistics] [--lang LANG]
    ```
    * Notes:
    To drive code property graph construction, you need to specify the path to retrieve
    code fragments (--src_path) and the path to store CPG results
    (--encode_path). Besides, for C programs, you should specify code clone or
    classification (--clone_classification).

* CPG-based neural network (CPGNN)

    * Show usage and help:

    ```bash
    python main_oj.py -h

    usage: driver [-h] [-l LOGGING] [--gpu_id GPU_ID] [--dataset DATASET]
              [--splitlabel] [--pretrain PRETRAIN] [--model_type MODEL_TYPE]
              [--adj_type ADJ_TYPE] [--epoch EPOCH] [--lr LR] [--regs [REGS]]
              [--opt_type OPT_TYPE] [--mess_dropout [MESS_DROPOUT]]
              [--early_stop] [--type_dim TYPE_DIM]
              [--word2vec_window WORD2VEC_WINDOW]
              [--word2vec_count WORD2VEC_COUNT]
              [--word2vec_worker WORD2VEC_WORKER] [--word2vec_save]
              [--embed_init EMBED_INIT] [--layer_size [LAYER_SIZE]]
              [--agg_type [AGG_TYPE]] [--save_model]
              [--classification_num CLASSIFICATION_NUM]
              [--clone_threshold CLONE_THRESHOLD]
              [--batch_size_clone BATCH_SIZE_CLONE]
              [--clone_test_unsupervised] [--clone_test_supervised]
              [--clone_val_size CLONE_VAL_SIZE]
              [--clone_test_size CLONE_TEST_SIZE] [--classification_test]
              [--class_val_size CLASS_VAL_SIZE]
              [--class_test_size CLASS_TEST_SIZE]
              [--batch_size_classification BATCH_SIZE_CLASSIFICATION]
              [--classification_visualization] [--report REPORT]
    ```
    * Notes:
    Simplicity is one of our guiding principles in designing CPGNN, where only
    two hyper-parameters --- the embedding size of type/token symbols
    (--type_dim) and the number of propagation iterations (--layer_size) --- are
    needed to be revised towards more effective directions. Note that you need
    to define two GPUs for training (e.g., --gpu_id 0,1).

## Dataset
Tailor has been evaluated on two public datasets: `OJClone` and `BigCloneBench`. 

* OJClone dataset
    * OJClone dataset is collected from a pedagogical online judge system, which
      contains 52,000 C programs belonging to 104 programming tasks. In our
      experiments, we use the first 15 tasks for code clone detection and all
      the tasks for source code classification. 

* BigCloneBench dataset
    * The BigCloneBench dataset is a popular code clone benchmark collected from
      25,000 Java software systems, which contains 6,000,000 clone pairs and
      260,000 non-clone pairs. Additionally, clone pairs can be categorized as
      `Type-1`, `Type-2`, `Strongly Type-3`, `Moderately Type-3`, and `Weakly
      Type-3/Type-4` according to line-level and token-level similarity. In our
      experiments, we sample 20,000 pairs for each clone types as positive
      samples and 20,000 non-clone pairs as negative samples. Note that the
      BigCloneBench dataset is not used in source code classification due to the
      lack of ground truth.

## Reproducibility
To demonstrate the reproducibility of experimental results reported in the paper
and facilitate future research, we provide the best parameter settings in the
[script](log/artifact_cpgnn.py), and show [our evaluation logs](log/README.md). Please note that
we also include the scripts and training logs for the ablation studies in the paper
(e.g., how different GNN variants affect Tailor?). 

Additionally, we include two examples of [code property graphs](cpg/README.md) build upon C and
Java programs.

* OJClone dataset
    * CPG Construction and Encoding for Code Clone Detection (~ 8 mins)
     ```bash
    # If you want to skip this procedure, you can download oj_clone_encoding.tar.gz from https://drive.google.com/file/d/1sQuMFwuelufxoP3_iAbpYO2rfdc3OzPU
    # Make sure you have decompressed datasets/ojclone.tar.gz
    cd cpg
    python driver.py --lang c --clone_classification clone --src_path ../datasets/ojclone --statistics --encoding --encode_path ../cpgnn/data/oj_clone_encoding
    ```
    * Code Clone Detection (~ 2 hours)

    ```bash
    # Make sure you have oj_clone_encoding under cpgnn/data
    cd cpgnn
    python main_oj.py --clone_test_supervised --epoch 30 --classification_num 15 --clone_threshold 0.5 --dataset oj_clone_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_clone 512 --gpu_id 0,1 --report clone_oj
    ```

    * CPG Construction and Encoding for Source Code Classification (~ 40 mins)
     ```bash
    # If you want to skip this procedure, you can download oj_classification_encoding.tar.gz from https://drive.google.com/file/d/1u9s4K43NluxFMVLKoqFoLVJIKRIANuA2
    # Make sure you have decompressed datasets/ojclassification.tar.gz
    cd cpg
    python driver.py --lang c --clone_classification classification --src_path ../datasets/ojclassification --statistics --encoding --encode_path ../cpgnn/data/oj_classification_encoding
    ```

    * Source Code Classification (~ 8 hours)

    ```bash
    # Make sure you have oj_classification_encoding under cpgnn/data.
    cd cpgnn
    python main_oj.py --classification_test --epoch 251 --classification_num 104 --dataset oj_classification_encoding --type_dim 16 --layer_size [32,32,32,32,32] --batch_size_classification 384 --gpu_id 0,1 --report classification_oj
    ```

* BigCloneBench dataset
    * CPG Construction and Encoding for Code Clone Detection (~ 30 mins)
    ```bash
    # If you want to skip this procedure, you can download bcb_clone_encoding.tar.gz from https://drive.google.com/file/d/1AQmGqxsMavWbd0fkphHRNr-nXic9wcj8
    # Make sure you have decompressed datasets/bigclonebench.tar.gz
    cd cpg
    python driver.py --lang java --src_path ../datasets/bigclonebench --statistics --encoding --encode_path ../cpgnn/data/bcb_clone_encoding
    ```
    * Code Clone Detection (~ 4 hours)

    ```bash
    # Make sure you have bcb_clone_encoding under Tailor/cpgnn/data
    cd cpgnn
    python main_bcb.py --clone_test_supervised --epoch 30 --clone_threshold 0.5 --dataset bcb_clone_encoding --type_dim 16 --layer_size [32,32,32,32] --batch_size_clone 384 --gpu_id 0,1 --report clone_bcb
    ```

# License
Tailor is released under the [GPL-3.0 license](https://www.gnu.org/licenses/gpl-3.0.html). 

# Publication
Please consider to cite our paper if you use the code in your research project.
```
@inproceedings{Tailor23ICSE,
  author    = {Jiahao Liu and
               Jun Zeng and
               Xia Wang and
               Zhenkai Liang},
  title     = {Learning Graph-based Code Representations for Source-level Functional Similarity Detection},
  booktitle = {ICSE},
  year      = {2023}
}
```

# Contact
Jun ZENG (junzeng@comp.nus.edu.sg) and Jiahao LIU (jiahao99@comp.nus.edu.sg)
