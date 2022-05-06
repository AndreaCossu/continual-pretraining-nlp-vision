
Continual finetuning hyperparameters are the one specified in the paper and are the same for all experiments.  
In particular: 20 epochs with early stopping, 
learning rate of 1e-5 and batch size of 10 without weight decay. 

Continual pretraining hyperparameters are the one specified in the paper and are the same for all experiments.  
In particular: 30 epochs with early stopping, 
learning rate of 5e-5 and batch size of 25 without weight decay. 

For the linear evaluation experiments we used learning rate of 1e-3 for 100 epochs with early stopping and batch size of 64.

NLP experiments were not very sensitive with respect to the change in hyperparameters values.