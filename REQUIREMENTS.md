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