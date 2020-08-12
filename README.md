# cnn_crash_course
Lunch n' Learn Jupyter notebook - Convolutional Neural Network crash course.

# Installation

* Install Donker (https://docs.docker.com/get-docker/)
* Clone and pull this repo.
* Run the following in your local repo:
```docker run -d -p 8888:8888 -v $(pwd):/srv gw000/keras-full``` Password is 'keras' (Uses this container: https://hub.docker.com/r/gw000/keras-full/)

In your browser, go to http://localhost:8888/notebooks/train.ipynb

To use Tensorboard, go to http://localhost:8888/, check-select the 'logs' directory and click the 'Tensorboard' button.  This should start a tensorboard instance using an tensorboard events files located in the logs directory.
Check out the 'usage' section [here](https://github.com/lspvic/jupyter_tensorboard).

## PlaidML

PlaidML makes stuff run much faster on a macbook.  If you want to use PlaidML, you can't use the donker image, though.  Instead, install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and in the local repo, run:

`conda env create -f environment.yml`

You can then run the example in `/train.py`
