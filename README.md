# Accurate and interpretable evaluation of surgical skills from kinematic data using fully convolutional neural networks
This is the companion repository for [our paper](https://link.springer.com/article/10.1007/s11548-019-02039-4) titled "Accurate and interpretable evaluation of surgical skills from kinematic data using fully convolutional neural networks" published in the [International Journal of Computer Assisted Radiology and Surgery](https://link.springer.com/journal/11548) - Special Issue of [MICCAI 2018](https://www.springer.com/gp/book/9783030009335), also available on ArXiv [[TODO]]. 

## Architecture
![architecture fcn](https://github.com/hfawaz/ijcars19/blob/master/archi-1.png)

## Data 
You will need the [JIGSAWS dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) to re-run the experiments of the paper.

## Requirements
You will need to install the following packages present in the [requirements.txt](https://github.com/hfawaz/ijcars19/blob/master/requirements.txt) file. 

## Running the code
### Classification
You will first need to run the following: ```python3 classification.py```. 

Then compile the results by running the following: ```python3 classification.py results```.

Finally, to visualize the class activation map, you will need to run the following: ```python3 classification.py cas```.

Here is an example of the class activation map for the classification task.

Expert             |  Novice
:-------------------------:|:-------------------------:
![class-st-e002](https://github.com/hfawaz/ijcars19/blob/master/class-st-e002-1.png)  |  ![class-st-h004](https://github.com/hfawaz/ijcars19/blob/master/class-st-h004-1.png)

### Regression
You will first need to run the following: ```python3 regression.py```.

Then compile the results by running the following: ```python3 regression.py results```.

Finally, to visualize the class activation map, you will need to run the following: ```python3 regression.py cas```.

Here is an example of the class activation map for the regression task.

Suture/needle handling             |  Quality of the final product
:-------------------------:|:-------------------------:
![reg-2-kt-e002](https://github.com/hfawaz/ijcars19/blob/master/reg-2-kt-e002-1.png)  |  ![reg-6-kt-e002](https://github.com/hfawaz/ijcars19/blob/master/reg-6-kt-e002-1.png)

## Reference
If you re-use this work, please cite:

```
@Article{ismailfawaz2019accurate,
  author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  title                    = {Accurate and interpretable evaluation of surgical skills from kinematic data using fully convolutional neural networks},
  journal                  = {International Journal of Computer Assisted Radiology and Surgery},
  year                     = {2019}
```
