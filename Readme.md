# P-DroNet Powerline Tracking

Link repo: https://rtgit.rta.vn/rtr_huylda/powerline-tracking

<p>This repository contains the code used to train and evaluate P-DroNet based on PyTorch.</p>

## Model

![architecture](./imgs/architecture.png)

## Data

<p>In order to learn tracking angles, Unreal Engine dataset has been used. We additionally recorded an virtual pole dataset to learn the angle of wire on powerline by riding a drone around the the transmitter power in Real-time Robotics Viet Nam company.</p>

![IMG01_0502](./imgs/IMG01_0502.png)

## Software requirements

This code has been tested on Ubuntu 18.04, and on Python 3.6.
Dependencies:

> - Install and build Python 3.6
> - PyTorch==1.7.0
>   > - `pip3 install torch==1.7.0 torchvision==0.8.0`
> - Tensorflow==1.5.0
>   > - `python3 -m pip install tensorflow-gpu==1.5.0`
> - Keras==2.1.4
>   > - `pip3 install keras==2.1.4`
> - Cython
>   > - `pip install Cython`
> - NumPy==1.17.0
>   > - `pip install numpy==1.17.0`
> - skikit-learn==0.18.1
>   > - `pip3 install skikit-learn==0.18.1`
> - scipy==1.0.0
>   > - `pip3 install scipy==1.0.0`
> - OpenCV==4.1.0
>   > - `build and make`

## Data preparation

#### Tracking angle and Powerline detection (href: /home/rtr/Powerline_Tracking/datasets/)

The final structure of the steering dataset should look like this:

```
training/
    Image01_Tracking/*
        images/
        tracking_angle.txt
    Image02_Tracking/
    Image03_Tracking/
    Image01_Powerline/*
        images/
        label.txt
    Image02_Powerline/
    Image03_Powerline/
    ...
validation/
    Image10_Tracking/
    Image10_Powerline/
    ...
testing/
    Image11_Tracking/
    Image11_Powerline/
    ...
```

## Training P-DroNet from scratch

> - `sudo python3 Training_Tracking_Gimbal.py`

## Evaluating P-DroNet

> - `sudo python3 evaluation.py`

## Plot results

> - `sudo python3 plot_results.py`

## Plot loss

> - `sudo python3 plot_loss.py`

# References

> https://github.com/uzh-rpg/rpg_public_dronet
