Supplementary code for the paper submitted to ICDL 2022 "Feedback-Driven Incremental Imitation Learning Using Sequential VAE". [paper](https://paperhost.org/proceedings/ras/ICDL22/files/0072.pdf)

Link to the [video](https://www.youtube.com/watch?v=DEfvUAwfLpc&t=6s)

# Pepper Imitation VAE

The goal of this project is to create a few-shot incremental learning framework with Pepper robot in which it can learn to perform arm movements based on human demonstrations and natural language descriptions/labels. The model used is a sequential VAE.

The library used for pose detection is a Pytorch implementation of Openpose ([link here](https://github.com/Hzzone/pytorch-openpose)).
We detect positions of human arms and count angles between each two joints on each arm. We also detect whether the hand is closed or open (Pepper cannot move individual fingers). These values are then used for robot control.   
For Pepper control, we use a modification of the [Pepper Controller](https://github.com/incognite-lab/Pepper-Controller).


## Requirements & dataset

We recommend to install a conda env from environment.yml:


`conda env create -f environment.yml`

`conda activate pepper-pose`


You can use our dataset provided in the ./dataset folder. We also provide code to record your own data using Pepper robot. 


### Record your own dataset

First, download the OpenPose body estimation model (for your own data collection):

`cd src`

`./download_openpose_models.sh`

Next, turn on your Pepper robot and get it's IP address. Then run:

`python run.py -m record -ip 0.0.0.0`  (replace with the IP address of your robot)

## Run code

Before training, edit "config.yml" according to your needs. 


You can then either only train (mode "train"), evaluate on the robot (mode "eval") or train a model and evaluate afterwards (mode "train_eval"). 


`cd src`

`python run.py --mode train_eval --robot_ip 0.0.0.0`
