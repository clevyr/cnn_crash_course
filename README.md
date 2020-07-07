# cnn_crash_course
Lunch n' Learn Jupyter notebook - Convolutional Neural Network crash course.

#Installation

Install Donker (https://docs.docker.com/get-docker/)

Use this Docker Container 

Run the following in your local repo:
```docker run -d -p 8888:8888 -v $(pwd):/srv gw000/keras-full``` (Uses this container: https://docs.docker.com/get-docker/)

In your browser, go to http://localhost:8888/notebooks/train.ipynb

To use Tensorboard, go to http://localhost:8888/, check-select the 'logs' directory and click the 'Tensorboard' button.  This should start a tensorboard instance using an tensorboard events files located in the logs directory.
Check out the 'usage' section [here](https://github.com/lspvic/jupyter_tensorboard).
