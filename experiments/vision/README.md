Each file provides hyperparameter for one experiment.
The model name is either "beit", "vit" or "resnet". The "linear" suffix means that the experiment
refers to the linear evaluation setup. The "joint" refers to joint training performance, 
while "cl" refers to the performance after each step of continual pretraining.

The continual pretraining hyperparameters are the ones used in the "modelname"_"cl".json experiments.