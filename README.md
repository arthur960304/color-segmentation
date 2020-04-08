# Color Segmentation

This project aims to train a probabilistic color model from image data and use it to segment unseen images, detect stop signs in an image, and draw bounding boxes around them. Logistic Model and Simple Gaussian Model are used in this projects for comparison.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Please review [requirements.txt](https://github.com/arthur960304/color-segmentation/blob/master/requirements.txt).
Download the training data [here](https://drive.google.com/open?id=158j9YPU_k0C2HijSh_9kc8x8VUMwF2ZU).

## Code organization

    .
    ├── stop_sign_detector.py   # Main stop sign detector file
    ├── gaussian                # Scripts for Gaussian
    │   ├── gaussian_model.py   # Implement gaussian model
    │   └── guassian_param.py   # Calculate mean and covariance for every class
    ├── logistic		        # Scripts for Logistic Regression
    │   ├── logistic_model.py   # Implement logistic regression model
    │   └── train_logistic.py   # Train the logistic regression model
    ├── labeltool.py            # hand-labeling tool, created by Chun-Nien Chan
    └── README.txt

## Running the tests

### Steps

1. Run the command `python stop_sign_detector.py` and the resulting images will display.

## Implementations

* See the [report](https://github.com/arthur960304/color-segmentation/blob/master/report/report.pdf) for detailed implementations.

## Results
![Example Result](https://github.com/arthur960304/color-segmentation/blob/master/results/log_bbox_1.png)


## Authors

* **Arthur Hsieh** - *Initial work* - [arthur960304](https://github.com/arthur960304)
