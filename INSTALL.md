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