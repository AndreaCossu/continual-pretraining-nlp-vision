# Continual Pre-Training Mitigates Forgetting in Language and Vision

## Prerequisites
* Install PyTorch (GPU support strongly recommended)
* Install [Avalanche library](https://github.com/ContinualAI/avalanche/) for continual learning with `pip install avalanche-lib`. Check that version is 0.1.0 (beta). 
* We rely on Huggingface for NLP experiments.
* We rely on [`torch_cka`](https://pypi.org/project/torch-cka/) package to compute CKA
* Specify data paths in `utils.py` file. This is used to locate data across all experiments.

We provide an environment file `env.yml` with which you can build the conda environment we used in the paper.

## NLP environment
[Download](https://drive.google.com/file/d/18gGyDFJuYkrePX8GOvK3nkqeaI6YWK23/view?usp=sharing) the preprocessed version of the NLP benchmarks.
The total size after unpacking should be around `21GB`.
This includes tokenized datasets in different versions: for Bert, Roberta and Roberta with expanding vocabulary. 
In the latter case, you can find one tokenized version of each dataset per experience. 
Put the downloaded file under the same directory. Specify these paths in the `utils.py`.

When pretraining or finetuning the original pretrained model, you can pass the Huggingface modelname (e.g., `roberta-base`) to the `modelname` parameter of each script. When finetuning the pretrained model, you should pass
the path to the folder where the pretrained model has been saved after pretraining.

The script `pretraining-nlp.py` applies a *single* step of pretraining/downstream finetuning. The `task_type` parameter distinguishes between the two tasks.. 

The script `continual-pretraining-nlp.py` applies continual pretraining/downstream finetuning (on both scientific abstracts and proxy datasets). The `task_type` parameter distinguishes between the possible tasks.

### SentEval experiments
Clone the [official SentEval repository](https://github.com/facebookresearch/SentEval) and put it in the main project directory under the name `SentEval`.  
Run the `senteval.py` script.

### Build NLP benchmarks from original data
If you do not want to use the version of the dataset we prepared, you can also start from the original data and preprocess them.
Link to original datasets can be found in the appendix of our paper.

In particular, `tokenize_and_save.py` prepare the datasets for offline training (no continual learning phase). This is used to study the effect of a single pretraining step on the entire dataset of abstract classification.
The script takes the datasets and build the subsampled version we used in our experiments. 
You need to change the path to the data folders and you can change few options in the first rows, like tokenizer type used in preprocessing and whether to use expanding vocabulary for Roberta or not.
The `pretrain` and `finetuning` nomenclature for the scientific abstracts preprocessing distinguish between the splits used for pretraining and the splits used for finetuning.

The script `tokenize_and_save_cl.py` adopts the same approach but for the continual pretraining stream. It produces a tokenized version of the benchmarks for each experience.

## Computer Vision environment
CORe50 is available for automatic download through Avalanche. iNaturalist is available through TorchVision (follow instructions from TorchVision API).

The script `continual-pretraining-cv-supervised.py` performs continual pretraining with supervised image classification protocol.

The script `continual-pretraining-cv-unsupervised.py` performs continual pretraining with unsupervised masked image modelling protocol.

## CKA analysis
The `cka.py` script produces the CKA plots representing similarity across each model layer. 
The script compares the original pretrained model with the continuously pretrained model.
We used the default hyperparameters of the script.

## Traditional continual learning scenario for NLP environment
We provide a script (`traditional_cl_nlp.py`) to run the NLP experiments in the traditional continual learning scenario (no continual pretraining), where a model is continuoulsy finetuned on the scientific abstracts dataset.
The model is tested on all the experiences of scientific abstracts to measure forgetting. Catastrophic forgetting happens as expected here.

## Utils file
The `utils.py` file contains a broad set of utilities used in the experiments. 

Importantly, there you can find the list of abstract classes used in our experiments and the 
list of tokens added to Roberta when its vocabulary is expanding. The list is specified both for the single step pretraining and for the continual pretraining experiment (one list of tokens per experience).

All the custom Avalanche strategies (e.g., used to adapt Huggingface for Avalanche) are also defined there. 

For iNaturalist, we provide the indices of the few patterns we removed because they did not match the default image format of the dataset.

## Reproducing experiments
The folder `experiments/vision` provides files showing the hyperparameter configurations used in our experiments. You can reproduce 
results by running the main script of the experiment you want with the corresponding hyperparameters values.
For NLP experiments hyperparameters are described in the README of the corresponding folder `experiments/nlp`.

## Jupyter notebook
We provide a Jupyter notebook to replicate our exploratory analysis of the NLP environment, including the analysis on the number of
abstracts per class and the most frequent tokens present in the dataset with respect to Roberta tokenizer. 
In particular, we used this last information to select which tokens to add to Roberta when its vocabulary was expanding. 
The Jupyter notebook is provided *as is*, since we used it for data exploration and not for the actual execution of experiments.
