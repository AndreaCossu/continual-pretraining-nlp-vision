from copy import deepcopy
import os
import random
import math
import pickle
import json
import avalanche as avl
from PIL import Image
from avalanche.benchmarks.utils.dataset_definitions import IDatasetWithTargets
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheSubset, AvalancheDatasetType, AvalancheConcatDataset
import numpy as np
import torch
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.benchmarks.utils.dataset_utils import TupleTLabel
from torchvision.transforms import InterpolationMode, autoaugment
from transformers.trainer import Trainer
from tqdm import tqdm
from pandas import read_csv, concat
from sklearn.model_selection import train_test_split
from datasets import Features, Value, ClassLabel, Dataset, Split, load_metric, concatenate_datasets
from transformers import TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback, AutoTokenizer, \
    BeitFeatureExtractor, BeitForImageClassification, BeitForMaskedImageModeling
from torchvision import transforms
from dall_e import load_model
from dall_e.utils import map_pixels

base_data = '/disk3/cossu'
base_save = '/disk2/cossu'


root_path = os.path.join(base_data, 'arxiv_archive-master/processed_data/20200101/per_year')
save_path = os.path.join(base_data, 'arxiv_archive-master/saved')
save_path_small = os.path.join(base_data, 'arxiv_small')
save_path_generic = os.path.join(base_data, 'data', 'emotion')
save_path_qnli = os.path.join(base_data, 'data', 'glue', 'qnli', '1.0.0')
cache_dir = os.path.join(base_save, 'huggingface')
wb_save = os.path.join(base_data, 'wandb')
log_dir = os.path.join(base_data, 'logs')

f1 = load_metric("f1", cache_dir=cache_dir)
accuracy = load_metric('accuracy', cache_dir=cache_dir)

classes = ['acc-phys', 'cmp-lg', 'cs.CE', 'cs.GT', 'cs.PL', 'funct-an', 'math.DS', 'math.OC', 'nucl-ex',
           'physics.gen-ph', 'q-bio.MN', 'q-fin.ST', 'adap-org', 'comp-gas', 'cs.CG', 'cs.HC', 'cs.RO', 'gr-qc',
           'math.FA', 'math-ph', 'nucl-th', 'physics.geo-ph', 'q-bio.NC', 'q-fin.TR', 'alg-geom', 'cond-mat.dis-nn',
           'cs.CL', 'cs.IR', 'cs.SC', 'hep-ex', 'math.GM', 'math.PR', 'patt-sol', 'physics.hist-ph', 'q-bio.OT',
           'quant-ph', 'ao-sci', 'cond-mat.mes-hall', 'cs.CR', 'cs.IT', 'cs.SD', 'hep-lat', 'math.GN', 'math.QA',
           'physics.acc-ph', 'physics.ins-det', 'q-bio.PE', 'solv-int', 'astro-ph.CO', 'cond-mat.mtrl-sci', 'cs.CV',
           'cs.LG', 'cs.SE', 'hep-ph', 'math.GR', 'math.RA', 'physics.ao-ph', 'physics.med-ph', 'q-bio.QM', 'stat.AP',
           'astro-ph.EP', 'cond-mat.other', 'cs.CY', 'cs.LO', 'cs.SI', 'hep-th', 'math.GT', 'math.RT', 'physics.app-ph',
           'physics.optics', 'q-bio.SC', 'stat.CO', 'astro-ph.GA', 'cond-mat.quant-gas', 'cs.DB', 'cs.MA', 'cs.SY',
           'math.AC', 'math.HO', 'math.SG', 'physics.atm-clus', 'physics.plasm-ph', 'q-bio.TO', 'stat.ME',
           'astro-ph.HE', 'cond-mat.soft', 'cs.DC', 'cs.MM', 'dg-ga', 'math.AG', 'math.IT', 'math.SP',
           'physics.atom-ph', 'physics.pop-ph', 'q-bio', 'stat.ML', 'astro-ph.IM', 'cond-mat.stat-mech', 'cs.DL',
           'cs.MS', 'econ.EM', 'math.AP', 'math.KT', 'math.ST', 'physics.bio-ph', 'physics.soc-ph', 'q-fin.CP',
           'stat.OT', 'astro-ph.SR', 'cond-mat.str-el', 'cs.DM', 'cs.NA', 'econ.GN', 'math.AT', 'math.LO', 'mtrl-th',
           'physics.chem-ph', 'physics.space-ph', 'q-fin.EC', 'stat.TH', 'astro-ph', 'cond-mat.supr-con', 'cs.DS',
           'cs.NE', 'econ.TH', 'math.CA', 'math.MG', 'nlin.AO', 'physics.class-ph', 'plasm-ph', 'q-fin.GN', 'supr-con',
           'atom-ph', 'cond-mat', 'cs.ET', 'cs.NI', 'eess.AS', 'math.CO', 'math.MP', 'nlin.CD', 'physics.comp-ph',
           'q-alg', 'q-fin.MF', 'bayes-an', 'cs.AI', 'cs.FL', 'cs.OH', 'eess.IV', 'math.CT', 'math.NA', 'nlin.CG',
           'physics.data-an', 'q-bio.BM', 'q-fin.PM', 'chao-dyn', 'cs.AR', 'cs.GL', 'cs.OS', 'eess.SP', 'math.CV',
           'math.NT', 'nlin.PS', 'physics.ed-ph', 'q-bio.CB', 'q-fin.PR', 'chem-ph', 'cs.CC', 'cs.GR', 'cs.PF',
           'eess.SY', 'math.DG', 'math.OA', 'nlin.SI', 'physics.flu-dyn', 'q-bio.GN', 'q-fin.RM']

filtered_classes = [53, 120, 65, 35, 37, 17, 49, 109, 97, 108]
# ['hep-ph', 'astro-ph', 'hep-th', 'quant-ph', 'cond-mat.mes-hall', 'gr-qc', 'cond-mat.mtrl-sci',
# 'cond-mat.str-el', 'cond-mat.stat-mech', 'astro-ph.SR']

additional_tokens_word_cl = [
    ['observations', 'distribution', 'temperature', 'luminosity', 'parameters', 'evolution', 'structure', 'galaxies',
     'emission', 'neutrino', 'observed', 'magnetic', 'clusters', 'spectrum', 'velocity', 'galactic', 'redshift',
     'obtained', 'symmetry', 'spectral', 'cluster', 'sources', 'optical', 'spectra', 'discuss', 'galaxy', 'masses',
     'theory', 'higgs', 'gamma', 'quark', 'decay', 'gauge', 'solar', 'flux', 'qcd', 'gev', '$\\', '}$'],
    ['supersymmetric', 'entanglement', 'hamiltonian', 'measurement', 'interaction', 'dimensions', 'particular',
     'conditions', 'operators', 'equations', 'functions', 'conformal', 'entangled', 'potential', 'solutions',
     'classical', 'boundary', 'equation', 'possible', 'theories', 'approach', 'particle', 'solution', 'coupling',
     'dynamics', 'algebra', 'systems', 'quantum', 'entropy', 'scalar', 'finite', 'photon', 'cavity', 'qubits', 'matrix',
     'branes', 'scheme', 'qubit', 'brane', 'dual', '$-', ')$'],
    ['gravitational', 'cosmological', 'topological', 'conductance', 'scattering', 'transition', 'relativity',
     'electronic', 'spacetime', 'numerical', 'transport', 'tunneling', 'parameter', 'electrons', 'einstein', 'constant',
     'electron', 'universe', 'graphene', 'momentum', 'horizon', 'coupled', 'regime', 'tensor', 'metric', 'modes',
     'dirac', 'noise'],
    ['antiferromagnetic', 'magnetization', 'ferromagnetic', 'measurements', 'experimental', 'calculations',
     'temperatures', 'conductivity', 'interactions', 'correlation', 'structures', 'dependence', 'insulator',
     'materials', 'exchange', 'orbital', 'studied', 'crystal', 'thermal', 'lattice', 'doping', 'phases', 'phonon',
     'strain', 'kondo', 'films', 'fermi', 'bulk', '_2', '_3'],
    ['simulations', 'equilibrium', 'accretion', 'particles', 'diffusion', 'rotation', 'scaling', 'coronal', 'dwarf',
     'flare']]



additional_tokens_word = ['antiferromagnetic', 'renormalization', 'superconducting', 'electromagnetic', 'supersymmetric',
                     'experimentally', 'characteristic', 'representation', 'configurations', 'gravitational',
                     'corresponding', 'approximation', 'ferromagnetic', 'magnetization', 'distributions',
                     'significantly', 'thermodynamic', 'understanding', 'contributions', 'characterized',
                     'perturbations', 'configuration', 'spectroscopic', 'observational', 'distribution', 'observations',
                     'measurements', 'interactions', 'experimental', 'entanglement', 'calculations', 'fluctuations',
                     'cosmological', 'polarization', 'temperatures', 'correlations', 'investigated', 'relativistic',
                     'applications', 'contribution', 'oscillations', 'conductivity', 'respectively', 'spectroscopy',
                     'perturbation', 'differential', 'intermediate', 'perturbative', 'temperature', 'interaction',
                     'topological', 'simulations', 'investigate', 'correlation', 'theoretical', 'equilibrium',
                     'demonstrate', 'experiments', 'transitions', 'hamiltonian', 'measurement', 'corrections',
                     'constraints', 'probability', 'interacting', 'generalized', 'predictions', 'fundamental',
                     'statistical', 'excitations', 'calculation', 'possibility', 'furthermore', 'conductance',
                     'instability', 'frequencies', 'metallicity', 'observation', 'anisotropic', 'numerically',
                     'observables', 'variability', 'transition', 'parameters', 'scattering', 'particular', 'electronic',
                     'conditions', 'consistent', 'dependence', 'structures', 'dimensions', 'considered', 'calculated',
                     'determined', 'absorption', 'luminosity', 'relaxation', 'components', 'previously', 'anisotropy',
                     'analytical', 'additional', 'excitation', 'transverse', 'correlated', 'comparison', 'dispersion',
                     'simulation', 'continuous', 'relativity', 'mechanical', 'techniques', 'amplitudes', 'difference',
                     'variations', 'experiment', 'asymptotic', 'structural', 'abundances', 'statistics', 'introduced',
                     'stochastic', 'symmetries', 'approaches', 'structure', 'potential', 'evolution', 'equations',
                     'classical', 'solutions', 'parameter', 'particles', 'transport', 'functions', 'numerical',
                     'discussed', 'dynamical', 'materials', 'processes', 'electrons', 'molecular', 'mechanism',
                     'agreement', 'presented', 'radiation', 'expansion', 'amplitude', 'spacetime', 'accretion',
                     'determine', 'calculate', 'diffusion', 'symmetric', 'magnitude', 'invariant', 'formalism',
                     'resulting', 'detection', 'arbitrary', 'operators', 'resonance', 'nonlinear', 'technique',
                     'insulator', 'conformal', 'tunneling', 'predicted', 'continuum', 'performed', 'algorithm',
                     'therefore', 'stability', 'abundance', 'variables', 'couplings', 'curvature', 'anomalous',
                     'mechanics', 'principle', 'increases', 'phenomena', 'behaviour', 'telescope', 'polarized',
                     'radiative', 'localized', 'influence', 'introduce', 'entangled', 'inflation', 'intrinsic',
                     'densities', 'magnetic', 'electron', 'observed', 'emission', 'coupling', 'symmetry', 'obtained',
                     'dynamics', 'particle', 'equation', 'approach', 'spectrum', 'graphene', 'galaxies', 'possible',
                     'velocity', 'spectral', 'presence', 'constant', 'theories', 'momentum', 'solution', 'boundary',
                     'clusters', 'neutrino', 'proposed', 'universe', 'recently', 'rotation', 'compared', 'strongly',
                     'previous', 'infrared', 'exchange', 'measured', 'describe', 'einstein', 'provides', 'geometry',
                     'extended', 'galactic', 'energies', 'addition', 'detected', 'increase', 'disorder', 'elements',
                     'scenario', 'fermions', 'coherent', 'estimate', 'detailed', 'contrast', 'rotating', 'metallic',
                     'redshift', 'identify', 'fraction', 'periodic', 'binaries', 'enhanced', 'suggests', 'gaussian',
                     'moreover', 'indicate', 'analyzed', 'exhibits', 'directly', 'hydrogen', 'infinite', 'profiles',
                     'surfaces', 'resolved', 'collapse', 'combined', 'impurity', 'examples', 'explicit', 'quantum',
                     'systems', 'optical', 'lattice', 'discuss', 'thermal', 'spectra', 'cluster', 'however', 'studied',
                     'entropy', 'provide', 'several', 'orbital', 'physics', 'sources', 'coupled', 'regions', 'scaling',
                     'various', 'applied', 'methods', 'further', 'spatial', 'propose', 'neutron', 'finally', 'studies',
                     'crystal', 'measure', 'horizon', 'diagram', 'analyze', 'compare', 'exhibit', 'compute', 'compact',
                     'fermion', 'degrees', 'explain', 'depends', 'coulomb', 'reduced', 'smaller', 'becomes', 'coronal',
                     'samples', 'observe', 'minimal', 'photons', 'kinetic', 'formula', 'appears', 'typical', 'towards',
                     'theory', 'finite', 'scalar', 'photon', 'regime', 'phases', 'matrix', 'obtain', 'masses', 'galaxy',
                     'derive', 'tensor', 'larger', 'vacuum', 'phonon', 'chiral', 'scheme', 'metric', 'scales', 'mixing',
                     'radial', 'signal', 'survey', 'plasma', 'cosmic', 'strain', 'cavity', 'dwarfs', 'layers', 'cannot',
                     'latter', 'doping', 'beyond', 'curves', 'reveal', 'appear', 'decays', 'moment', 'landau', 'occurs',
                     'solar', 'gauge', 'gamma', 'modes', 'decay', 'noise', 'higgs', 'fermi', 'ratio', 'exact', 'delta',
                     'quark', 'leads', 'dirac', 'atoms', 'qubit', 'sigma', 'quasi', 'dwarf', 'brane', 'films', 'monte',
                     'carlo', 'means', 'boson', 'omega', 'novel', 'basis', 'spins', 'fluid', 'gives', 'probe', 'kondo',
                     'giant', 'flare', 'doped', 'argue', 'disks', 'pairs', 'flux', 'bulk', 'dual', '^{-', 'qcd', 'tau',
                     '}$,', '+/-', '}$.', 'gev', 'ngc', '$\\', '}$', ')$', '$-', '_2', '2d', '_3', 'm_', '3d', '$)']


def remap_classes(example):
    example['primary_cat'] = filtered_classes.index(example['primary_cat'])
    return example


class CustomTrainer(Trainer):
    def __init__(self, *args, eval_device='cpu', **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_device = torch.device(eval_device)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """https://github.com/huggingface/transformers/issues/7232"""
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if prediction_loss_only:
            return loss, None, None
        else:
            try:
                return loss, logits.to(self.eval_device), labels.to(self.eval_device)
            except RuntimeError:
                print(loss, logits, labels)


def create_tokenizer(tokenizer_name, add_tokens=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    if add_tokens:
        new_tokens_count = tokenizer.add_tokens(additional_tokens_word)
        print(f"Added {new_tokens_count} tokens to the tokenizer. Total tokens to add were {len(additional_tokens_word)}.")
    return tokenizer


def create_tokenizer_cl(tokenizer_name, exp_id, add_tokens=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    if add_tokens:
        toadd = []
        for i in range(exp_id+1):
            toadd += additional_tokens_word_cl[i]
        new_tokens_count = tokenizer.add_tokens(toadd)
        print(f"Added {new_tokens_count} tokens to the tokenizer. Total tokens to add were {len(toadd)}.")
    return tokenizer


def filter_by_classes(dataset, classes_filter):
    return dataset.filter(lambda el: el['primary_cat'] in classes_filter)


def create_small_dataset(dataset, patterns_per_class=None, num_patterns=None,
                         patterns_per_class_per_test=0):
    assert patterns_per_class or num_patterns
    if patterns_per_class is None:
        return dataset.select(range(num_patterns))
    else:
        classes_list = list(set(dataset['primary_cat']))
        dts = []
        dts_test = []
        for class_id in classes_list:
            dts_cls = filter_by_classes(dataset, [class_id])
            assert len(dts_cls) > patterns_per_class_per_test + patterns_per_class
            dts.append(dts_cls.select(range(patterns_per_class)))  # take from beginning
            if patterns_per_class_per_test > 0:  # take from end
                dts_test.append(dts_cls.select(range(len(dts_cls) - patterns_per_class_per_test, len(dts_cls))))

        if patterns_per_class_per_test > 0:
            return concatenate_datasets(dts), concatenate_datasets(dts_test)
        else:
            return concatenate_datasets(dts)


def load_generic_finetuning(use_test=False, add_tokens=False, exp_id=None, use_bert=False):
    append_to_save_dir = f'{exp_id}_new_tokens' if add_tokens else ''
    if use_bert:
        append_to_save_dir += '_bert'
    tr_d = Dataset.load_from_disk(os.path.join(save_path_generic, 'train_tokenized'+append_to_save_dir))
    if use_test:
        ts_d = Dataset.load_from_disk(os.path.join(save_path_generic, 'test_tokenized'+append_to_save_dir))
    else:
        ts_d = Dataset.load_from_disk(os.path.join(save_path_generic, 'valid_tokenized'+append_to_save_dir))
    return tr_d.remove_columns(['text']), ts_d.remove_columns(['text'])


def load_qnli(add_tokens=False, exp_id=None, use_bert=False):
    append_to_save_dir = f'{exp_id}_new_tokens' if add_tokens else ''
    if use_bert:
        append_to_save_dir += '_bert'
    tr_d = Dataset.load_from_disk(os.path.join(save_path_qnli, 'train_tokenized'+append_to_save_dir))
    ts_d = Dataset.load_from_disk(os.path.join(save_path_qnli, 'valid_tokenized'+append_to_save_dir))

    return tr_d.remove_columns(['sentence', 'question', 'idx']).rename_column('label', 'labels'), \
           ts_d.remove_columns(['sentence', 'question', 'idx']).rename_column('label', 'labels')


def compute_metrics(predictions):
    logits, targets = predictions
    predicted_classes = np.argmax(logits, axis=-1)
    out = {}
    out['accuracy'] = accuracy.compute(predictions=predicted_classes, references=targets)['accuracy']
    out['f1'] = f1.compute(predictions=predicted_classes, references=targets, average='macro')['f1']   # per class average without weight
    return out


def pretrain_model(args, tr_d, ts_d, model, tokenizer, eval_strategy, log_strategy, eval_only=False,
                   run_name=None):
    """Do not compute metrics other than loss to save memory space.
    https://github.com/huggingface/transformers/issues/8143"""
    model.train()
    ckp = os.path.join(log_dir, "masked_checkpoint_dir")
    training_args = TrainingArguments(ckp,
                                      evaluation_strategy=eval_strategy, eval_steps=args.eval_every,
                                      logging_strategy=log_strategy, logging_steps=args.log_every,
                                      save_strategy=eval_strategy, save_steps=args.eval_every,
                                      num_train_epochs=args.epochs,
                                      per_device_train_batch_size=args.train_batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      learning_rate=args.lr,
                                      weight_decay=args.weight_decay,
                                      no_cuda=args.no_cuda,
                                      report_to=["wandb"],
                                      run_name=run_name,
                                      load_best_model_at_end=True,
                                      metric_for_best_model='eval_loss',
                                      greater_is_better=False)

    callbacks = []
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    trainer = CustomTrainer(model=model, args=training_args, train_dataset=tr_d,
                            eval_dataset=ts_d, data_collator=data_collator,
                            eval_device='cpu',
                            callbacks=callbacks)
    if not eval_only:
        trainer.train(resume_from_checkpoint=False)
    trainer.evaluate()


def finetune_model(args, tr_d, ts_d, model, eval_strategy, log_strategy, eval_only=False,
                   run_name=None):
    model.train()
    ckp = os.path.join(log_dir, "checkpoint_dir")
    training_args = TrainingArguments(ckp,
                                      evaluation_strategy=eval_strategy, eval_steps=args.eval_every,
                                      logging_strategy=log_strategy, logging_steps=args.log_every,
                                      save_strategy=eval_strategy, save_steps=args.eval_every,
                                      num_train_epochs=args.epochs,
                                      per_device_train_batch_size=args.train_batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      learning_rate=args.lr,
                                      weight_decay=args.weight_decay,
                                      no_cuda=args.no_cuda,
                                      report_to=["wandb"],
                                      run_name=run_name,
                                      load_best_model_at_end=True,
                                      metric_for_best_model='eval_loss',
                                      greater_is_better=False)

    callbacks = []
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0))
    trainer = CustomTrainer(model=model, args=training_args, train_dataset=tr_d,
                            eval_dataset=ts_d, compute_metrics=compute_metrics,
                            eval_device='cpu',
                            callbacks=callbacks)
    if not eval_only:
        trainer.train(resume_from_checkpoint=False)
    trainer.evaluate()


def preprocess_and_tokenize_by_task(preprocess=False, tokenize=False, tokenizer=None,
                                    tokenize_field="abstract", append_to_save_dir=''):
    assert preprocess or tokenize

    if preprocess:
        features = Features({tokenize_field: Value('string'),
                             'primary_cat': ClassLabel(num_classes=len(classes), names=classes),
                             'created': Value('int64')})
        datas = []
        class_to_idx = dict(zip(classes, list(range(len(classes)))))
        for year in list(range(1993, 2020)):
            data = read_csv(os.path.join(root_path, f'{year}.tsv'), sep='\t',
                            usecols=[tokenize_field, 'primary_cat', 'created'],
                            converters={'created': lambda el: int(el[:4]),
                                        'primary_cat': lambda el: int(class_to_idx[el])})
            datas.append(data)
        data_all = concat(datas)
        print("Read all data")
        data_all = data_all.groupby('primary_cat').filter(lambda el: len(el) > 1)
        train_data, test_data = train_test_split(data_all, test_size=0.15, shuffle=True,
                                                 stratify=data_all['primary_cat'])
        pretrain_train, finetune_train = train_test_split(train_data, test_size=0.5, shuffle=True,
                                                          stratify=train_data['primary_cat'])
        pretrain_test, finetune_test = train_test_split(test_data, test_size=0.5, shuffle=True,
                                                        stratify=test_data['primary_cat'])
        print("Split train test completed")
        pr_tr = Dataset.from_pandas(pretrain_train, features=features, split=Split.TRAIN)
        pr_ts = Dataset.from_pandas(pretrain_test, features=features, split=Split.TEST)
        f_tr = Dataset.from_pandas(finetune_train, features=features, split=Split.TRAIN)
        f_ts = Dataset.from_pandas(finetune_test, features=features, split=Split.TEST)
        pr_tr.save_to_disk(os.path.join(save_path, 'train', 'pretrain_task_filtered'))
        pr_ts.save_to_disk(os.path.join(save_path, 'test', 'pretrain_task_filtered'))
        f_tr.save_to_disk(os.path.join(save_path, 'train', 'finetuning_task_filtered'))
        f_ts.save_to_disk(os.path.join(save_path, 'test', 'finetuning_task_filtered'))
        print("Saved split data")

    if tokenize:
        def tokenize(examples):
            return tokenizer(examples[tokenize_field], padding="max_length", truncation=True)

        pr_tr = Dataset.load_from_disk(os.path.join(save_path, 'train', 'pretrain_task_filtered'))
        pr_ts = Dataset.load_from_disk(os.path.join(save_path, 'test', 'pretrain_task_filtered'))
        f_tr = Dataset.load_from_disk(os.path.join(save_path, 'train', 'finetuning_task_filtered'))
        f_ts = Dataset.load_from_disk(os.path.join(save_path, 'test', 'finetuning_task_filtered'))
        print("Loaded data")
        pr_tr_tk = pr_tr.map(tokenize, batched=True, batch_size=1000)
        pr_tr_tk.save_to_disk(os.path.join(save_path, 'train', 'tokenized', 'pretrain_task_filtered'+append_to_save_dir))
        pr_ts_tk = pr_ts.map(tokenize, batched=True, batch_size=1000)
        pr_ts_tk.save_to_disk(os.path.join(save_path, 'test', 'tokenized', 'pretrain_task_filtered'+append_to_save_dir))
        f_tr_tk = f_tr.map(tokenize, batched=True, batch_size=1000)
        f_tr_tk.save_to_disk(os.path.join(save_path, 'train', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))
        f_ts_tk = f_ts.map(tokenize, batched=True, batch_size=1000)
        f_ts_tk.save_to_disk(os.path.join(save_path, 'test', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))
        print("Tokenization and saving completed")


class HGAvalancheDataset(AvalancheDataset):
    def _process_pattern(self, element, idx: int):
        pattern = element['input_ids']
        label = element['labels']

        pattern, label = self._apply_transforms(pattern, label)

        return TupleTLabel((pattern, label, element['attention_mask'], element['token_type_ids'],
                            self.targets_task_labels[idx]))


def create_abstracts_avalanche_benchmark(experiences, add_tokens=False, test_on_test=False):
    train_exps, test_exps = [], []
    for exp_id, classes in enumerate(experiences):
        append_to_save_dir = f'{exp_id}_new_tokens' if add_tokens and exp_id == 0 else f'{exp_id}'
        tr_d = Dataset.load_from_disk(os.path.join(save_path_small, 'train', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))

        if test_on_test:
            ts_d = Dataset.load_from_disk(os.path.join(save_path_small, 'test', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))
        else:
            ts_d = Dataset.load_from_disk(os.path.join(save_path_small, 'valid', 'tokenized', 'finetuning_task_filtered'+append_to_save_dir))

        tr_d = tr_d.map(remap_classes)
        ts_d = ts_d.map(remap_classes)

        tr_d = tr_d.remove_columns(['abstract', 'created']).rename_column('primary_cat', 'labels')
        ts_d = ts_d.remove_columns(['abstract', 'created']).rename_column('primary_cat', 'labels')
        tr_d.set_format(type="torch")
        ts_d.set_format(type="torch")

        tr_dataset = HGAvalancheDataset(tr_d, task_labels=0, dataset_type=avl.benchmarks.utils.AvalancheDatasetType.CLASSIFICATION,
                                        targets=tr_d['labels'])
        ts_dataset = HGAvalancheDataset(ts_d, task_labels=0, dataset_type=avl.benchmarks.utils.AvalancheDatasetType.CLASSIFICATION,
                                        targets=ts_d['labels'])

        train_exps.append(tr_dataset)
        test_exps.append(ts_dataset)

    benchmark = avl.benchmarks.dataset_benchmark(train_datasets=train_exps, test_datasets=test_exps,
                                                 dataset_type=avl.benchmarks.utils.AvalancheDatasetType.CLASSIFICATION)
    return benchmark


class HGFeatureExtractorBackbone(avl.models.FeatureExtractorBackbone):
    def forward(self, x):
        self.model(input_ids=x[0],
                   attention_mask=x[1],
                   token_type_ids=x[2])
        return self.output

    def resize_token_embeddings(self, x):
        self.model.resize_token_embeddings(x)


class HGStreamingLDA(avl.training.StreamingLDA):
    @property
    def mb_attention_mask(self):
        return self.mbatch[2]

    @property
    def mb_token_type_ids(self):
        return self.mbatch[3]

    def forward(self, return_features=False):
        self.model.eval()
        feat = self.model([self.mb_x, self.mb_attention_mask, self.mb_token_type_ids])
        out = self.predict(feat)
        if return_features:
            return out, feat
        else:
            return out


class HGBaseStrategy(avl.training.BaseStrategy):
    @property
    def mb_attention_mask(self):
        return self.mbatch[2]

    @property
    def mb_token_type_ids(self):
        return self.mbatch[3]

    def forward(self):
        out = self.model(input_ids=self.mb_x,
                         attention_mask=self.mb_attention_mask,
                         token_type_ids=self.mb_token_type_ids)
        return out.logits


class HGJointTraining(avl.training.JointTraining):
    @property
    def mb_attention_mask(self):
        return self.mbatch[2]

    @property
    def mb_token_type_ids(self):
        return self.mbatch[3]

    def forward(self):
        out = self.model(input_ids=self.mb_x,
                         attention_mask=self.mb_attention_mask,
                         token_type_ids=self.mb_token_type_ids)
        return out.logits


class ClassificationPresetTrain:
    """https://github.com/pytorch/vision/blob/main/references/classification/presets.py"""
    def __init__(
        self,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
        vit=False
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
        if not vit:
            trans.extend([transforms.Normalize(mean=mean, std=std), ])
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    """https://github.com/pytorch/vision/blob/main/references/classification/presets.py"""
    def __init__(
        self,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        vit=False
    ):

        trans = [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        if not vit:
            trans.extend([transforms.Normalize(mean=mean, std=std), ])

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


def get_remove_idxs(d):
    t = transforms.ToTensor()
    remove_idxs = []
    for i, el in enumerate(d):
        tel = t(el[0])
        if len(tel.size()) != 3 or tel.size(0) != 3:
            remove_idxs.append(i)
    return remove_idxs


class CustomExpDataTargets(IDatasetWithTargets):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        return self.images[index], self.targets[index]

    def __len__(self):
        return len(self.images)


def split_inaturalist(dataset, transforms_dict, test_size=0.25, target_type="super", in_memory=False):
    remove_idxs = [23513, 68871, 68888, 72691, 102203, 108354, 108358, 108385, 108440, 111947, 111948, 111958, 111967,
                   115891, 117965, 125014, 144830, 144834, 144839, 144846, 145495, 145496, 146083, 155378, 155379,
                   155381, 155382, 155383, 155384, 155386, 155387, 155388, 155389, 155390, 155392, 155393, 200064,
                   224063, 238914, 247984, 264555, 264557, 264561, 264562, 264565, 264566, 264567, 264569, 264570,
                   264571, 265398, 267506, 270337, 270342, 270345, 270350, 270351, 270352, 270354, 270363, 270364,
                   270365, 270368, 270377, 271500, 271951, 272514, 273215, 273290, 273294, 273295, 273296, 273297,
                   273302, 273320, 273392, 273554, 273558, 273566, 273707, 273737, 273805, 273865, 274083, 274173,
                   274532, 275093, 275095, 275098, 275100, 275103, 275105, 275106, 275107, 275108, 275110, 275111,
                   275112, 275116, 275171, 275175, 275177, 275178, 275179, 275180, 275182, 275183, 275184, 275185,
                   275187, 275190, 275191, 275192, 275193, 275197, 275206, 275211, 275214, 275220, 275221, 275223,
                   275224, 275227, 275228, 275229, 275230, 275233, 275236, 275237, 275238, 275301, 275397, 275419,
                   275434, 275498, 275574, 275580, 275581, 275605, 275668, 275799, 275824, 275928, 275937, 275939,
                   275940, 275942, 275944, 275945, 275946, 275948, 275949, 275950, 275951, 275952, 275954, 275955,
                   275957, 275958, 275959, 275960, 275961, 275962, 275963, 276329, 276556, 276742, 276743, 276745,
                   276746, 276747, 276749, 276750, 276754, 276756, 276768, 276779, 276781, 276831, 276832, 276834,
                   276835, 276837, 276838, 276840, 276842, 276844, 276846, 276847, 276849, 276850, 276851, 276852,
                   276853, 278278, 278315, 278441, 278444, 278451, 278581, 278627, 278866, 278930, 278957, 278963,
                   278966, 279007, 279034, 279050, 279060, 279073, 279094, 279128, 279139, 279181, 279297, 279573,
                   279685, 279686, 279690, 279696, 279706, 279708, 279709, 279710, 279712, 279714, 279715, 279716,
                   279717, 279718, 279719, 279722, 279724, 280828, 280878, 281020, 281274, 281338, 281344, 281346,
                   281558, 281563, 282062, 282526, 283876, 284824, 284852, 284853, 284857, 284860, 284861, 284863,
                   285438, 285441, 285442, 285444, 290245, 290246, 290247, 290249, 290250, 290252, 290256, 290257,
                   290259, 290261, 290263, 290264, 290266, 290267, 290268, 290269, 290271, 290274, 291206, 291209,
                   291211, 300037, 300040, 300053, 300054, 300056, 308274, 309049, 309507, 327167, 338681, 451448,
                   453391, 453397]
    if in_memory:
        images = []
        targets = []
        print("Start reading Inaturalist images.")
        for cat_id, fname in dataset.index:
            with Image.open(os.path.join(dataset.root, dataset.all_categories[cat_id], fname)) as m:
                images.append(m.copy())
                targets.append(dataset.categories_map[cat_id][target_type])
        print("Inaturalist images loaded in memory.")
        dts = AvalancheDataset(dataset=CustomExpDataTargets(images, targets), task_labels=0,
                               dataset_type=AvalancheDatasetType.CLASSIFICATION,
                               transform=None)
    else:
        try:
            with open(os.path.join(base_data, 'data', 'inaturalist_targets.pickle'), 'rb') as f:
                targets = pickle.load(f)
        except FileNotFoundError:
            print("Inaturalist targets not found. Creating them now.")
            targets = []
            for cat_id, fname in dataset.index:
                targets.append(dataset.categories_map[cat_id][target_type])
            with open(os.path.join(base_data, 'data', 'inaturalist_targets.pickle'), 'wb') as f:
                pickle.dump(targets, f)

        dts = AvalancheDataset(dataset, task_labels=0, targets=targets,
                               dataset_type=AvalancheDatasetType.CLASSIFICATION)

    train_idxs, test_idxs = train_test_split(range(len(targets)), stratify=targets, test_size=test_size, shuffle=True)
    for el in remove_idxs:
        if el in train_idxs:
            train_idxs.remove(el)
        if el in test_idxs:
            test_idxs.remove(el)
    if transforms_dict is not None:
        ttr = transforms_dict['train']
        tts = transforms_dict['val']
    else:
        ttr, tts = None, None
    dtrain = AvalancheSubset(dts, train_idxs, dataset_type=AvalancheDatasetType.CLASSIFICATION,
                             transform=ttr)
    dtest = AvalancheSubset(dts, test_idxs, dataset_type=AvalancheDatasetType.CLASSIFICATION,
                            transform=tts)
    return dtrain, dtest


def create_downstream_model(model, hidden_size, n_classes, device, vit=False, path=None):
    model_down = deepcopy(model)
    if path is not None:
        model_down.load_state_dict(torch.load(path))

    if vit:
        model_down.head = torch.nn.Linear(hidden_size, n_classes)
    else:
        model_down.fc = torch.nn.Linear(hidden_size, n_classes)
    return model_down.to(device)


class UnsupBEiTSeqCLF(torch.nn.Module):
    def __init__(self, num_classes, pretrained_path, pretrained_path_extractor, device='cpu'):
        super().__init__()
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(
            pretrained_path_extractor, cache_dir=os.path.join(base_data, 'data', 'beit'))
        self.vit = BeitForImageClassification.from_pretrained(
            pretrained_path, cache_dir=os.path.join(base_data, 'data', 'beit'),
            num_labels=num_classes)
        self.device = device

    def forward(self, x):
        if not isinstance(x, list):
            x = list(x.cpu())
        inputs = self.feature_extractor(images=x, return_tensors="pt")  # 1024 features
        outputs = self.vit(inputs['pixel_values'].to(self.device))
        return outputs.logits

    def save_pretrained(self, save_path):
        self.vit.save_pretrained(save_path)


class UnsupBEiTLM(torch.nn.Module):
    def __init__(self, pretrained_path, pretrained_path_extractor, device='cpu'):
        super().__init__()
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(
            pretrained_path_extractor, cache_dir=os.path.join(base_data, 'data', 'beit'))
        self.vit = BeitForMaskedImageModeling.from_pretrained(
            pretrained_path, cache_dir=os.path.join(base_data, 'data', 'beit'))
        self.device = device

        window_size = self.vit.beit.embeddings.patch_embeddings.patch_shape
        num_masking_patches = 75
        max_mask_patches_per_block = None
        min_mask_patches_per_block = 16

        # generating mask for the corresponding image
        self.mask_generator = MaskingGenerator(
                    window_size, num_masking_patches=num_masking_patches,
                    max_num_patches=max_mask_patches_per_block,
                    min_num_patches=min_mask_patches_per_block)

        self.encoder = load_model("https://cdn.openai.com/dall-e/encoder.pkl", device)
        self.transforms_beit = transforms.Compose([
                transforms.Resize((112, 112),
                transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
            ])

    def forward(self, x):
        if not isinstance(x, list):
            x = list(x.cpu())
        with torch.no_grad():
            """https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BEiT/Understanding_BeitForMaskedImageModeling.ipynb"""
            inputs = self.feature_extractor(images=x, return_tensors="pt")['pixel_values'].to(self.device)
            xtrans = torch.stack([self.transforms_beit(el) for el in x], dim=0).to(self.device)
            z_logits = self.encoder(map_pixels(xtrans))
            input_ids = torch.argmax(z_logits, dim=1).flatten(1)
            # bool_masked_pos = torch.from_numpy(self.mask_generator()).flatten().to(torch.bool)
            # bool_masked_pos = bool_masked_pos.unsqueeze(0).repeat(input_ids.size(0), 1)
            bool_masked_pos = torch.stack([torch.from_numpy(self.mask_generator()).flatten().to(torch.bool)
                                           for _ in range(input_ids.size(0))], dim=0)
            labels = input_ids[bool_masked_pos]

        outputs = self.vit(inputs, bool_masked_pos, labels=labels)  # batch_size, 1024, 512
        logits = outputs.logits[bool_masked_pos]
        return logits, outputs.loss

    def save_pretrained(self, save_path):
        self.vit.save_pretrained(save_path)


class UnsupBEiTNaive(avl.training.BaseStrategy):
    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        """ Data loader initialization.
        Called at the start of each learning experience after the dataset
        adaptation.
        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """
        self.dataloader = TaskBalancedDataLoader(
            self.adapted_dataset,
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            **kwargs)

    def make_eval_dataloader(self, num_workers=0, pin_memory=True,
                             **kwargs):
        """
        Initializes the eval data loader.
        :param num_workers: How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
            (default: 0).
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        :param kwargs:
        :return:
        """
        self.dataloader = torch.utils.data.DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=self.eval_mb_size,
            pin_memory=pin_memory,
            **kwargs)

    def _unpack_minibatch(self):
        assert len(self.mbatch) >= 3
        for i in range(1, len(self.mbatch)):
            self.mbatch[i] = self.mbatch[i].to(self.device)

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output, self.loss = self.forward()
            self._after_forward(**kwargs)

            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def eval_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output, self.loss = self.forward()
            self._after_eval_forward(**kwargs)
            self._after_eval_iteration(**kwargs)


class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


def freeze_half_model(model, freeze, bert=False):
    if not freeze:
        return

    modelname = 'bert' if bert else 'roberta'
    base_model = getattr(model, modelname)

    total_layers = len(base_model.encoder.layer)
    layers_to_freeze = total_layers // 2

    for p in base_model.embeddings.parameters():
        p.requires_grad_(False)

    for p in base_model.encoder.layer[:layers_to_freeze].parameters():
        p.requires_grad_(False)


def freeze_model_but_classifier(model, linear_eval, head_name):
    if not isinstance(head_name, (list, tuple)):
        head_name = [head_name]
    if not linear_eval:
        return
    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if any([el in n for el in head_name]):
            p.requires_grad_(True)


def json_save_dict(d, filepath):
    with open(filepath, 'w') as f:
        json.dump(d, f, indent=2)


def pickle_save_dict(d, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


class FixCORe50Dataset(avl.benchmarks.datasets.CORe50Dataset):
    def _load_metadata(self) -> bool:
        if self.mini:
            bp = "core50_32x32"
        else:
            bp = "core50_128x128"

        if not (self.root / bp).exists():
            return False

        if not (self.root / "batches_filelists").exists():
            return False

        with open(self.root / "paths.pkl", "rb") as f:
            self.train_test_paths = pickle.load(f)

        if self.verbose:
            print("Loading labels...")
        with open(self.root / "labels.pkl", "rb") as f:
            self.all_targets = pickle.load(f)
            self.train_test_targets = []
            for i in range(self._nbatch + 1):
                self.train_test_targets += self.all_targets[self._scen][
                    self._run
                ][i]

        if self.verbose:
            print("Loading LUP...")
        with open(self.root / "LUP.pkl", "rb") as f:
            self.LUP = pickle.load(f)

        if self.verbose:
            print("Loading labels names...")
        with open(self.root / "labels2names.pkl", "rb") as f:
            self.labels2names = pickle.load(f)

        self.idx_list = []
        if self.train:
            for i in range(self._nbatch):  # fix bug by removing + 1
                self.idx_list += self.LUP[self._scen][self._run][i]
        else:
            self.idx_list = self.LUP[self._scen][self._run][-1]

        self.paths = []
        self.targets = []

        for idx in self.idx_list:
            self.paths.append(self.train_test_paths[idx])
            div = 1
            if not self.object_level:
                div = 5
            self.targets.append(self.train_test_targets[idx] // div)

        with open(self.root / "labels2names.pkl", "rb") as f:
            self.labels2names = pickle.load(f)

        if not (self.root / "NIC_v2_79_cat").exists():
            self._create_cat_filelists()

        return True


nbatch = {
    "ni": 8,
    "nc": 9,
    "nic": 79,
    "nicv2_79": 79,
    "nicv2_196": 196,
    "nicv2_391": 391,
}

scen2dirs = {
    "ni": "batches_filelists/NI_inc/",
    "nc": "batches_filelists/NC_inc/",
    "nic": "batches_filelists/NIC_inc/",
    "nicv2_79": "NIC_v2_79/",
    "nicv2_196": "NIC_v2_196/",
    "nicv2_391": "NIC_v2_391/",
}
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_default_train_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.RandomHorizontalFlip(), normalize]
)
_default_eval_transform = transforms.Compose([transforms.ToTensor(), normalize])


def InMemoryCORe50(*, scenario: str = "nicv2_391", val_perc=0, run: int = 0, object_lvl: bool = True, mini: bool = False,
                   train_transform=_default_train_transform, eval_transform=_default_eval_transform, dataset_root=None,
                   only_test=False):
    assert 0 <= run <= 9, (
        "Pre-defined run of CORe50 are only 10. Indicate "
        "a number between 0 and 9."
    )
    assert scenario in nbatch.keys(), (
        "The selected scenario is note "
        "recognized: it should be 'ni', 'nc',"
        "'nic', 'nicv2_79', 'nicv2_196' or "
        "'nicv2_391'."
    )

    if dataset_root is None:
        dataset_root = default_dataset_location("core50")

    # Download the dataset and initialize filelists
    core_data = FixCORe50Dataset(root=dataset_root, mini=mini)

    root = core_data.root
    if mini:
        bp = "core50_32x32"
    else:
        bp = "core50_128x128"

    if object_lvl:
        suffix = "/"
    else:
        suffix = "_cat/"
    filelists_bp = scen2dirs[scenario][:-1] + suffix + "run" + str(run)
    train_failists_paths = []
    for batch_id in range(nbatch[scenario]):
        train_failists_paths.append(
            root
            / filelists_bp
            / ("train_batch_" + str(batch_id).zfill(2) + "_filelist.txt")
        )

    train_exps = []
    val_exps = []
    assert (not only_test) or val_perc == 0, "val_perc can be greater than zero only if only_test is False."

    if not only_test:
        for batch_id in range(nbatch[scenario]):
            print(f"Loading experience {batch_id} images in memory")
            dts = create_core50_exp_dataset(train_failists_paths[batch_id], root=root, bp=bp, val_perc=val_perc)
            if val_perc > 0:
                train_exps.append(dts[0])
                val_exps.append(dts[1])
            else:
                train_exps.append(dts)
            print("Done loading images")
    else:
        # fake train dataset
        train_exps = [AvalancheDataset(torch.utils.data.TensorDataset(torch.empty(3), torch.randint(0,2,(3,))))]
    print("Loading test images in memory")
    if val_perc == 0:
        test_set = create_core50_exp_dataset(root / filelists_bp / "test_filelist.txt", root=root, bp=bp)
    else:
        test_set = AvalancheConcatDataset(val_exps)
    print("Done loading images")
    benchmark = dataset_benchmark(train_datasets=train_exps,
                                  test_datasets=[test_set],
                                  complete_test_set_only=True,
                                  train_transform=train_transform, eval_transform=eval_transform,
                                  dataset_type=AvalancheDatasetType.CLASSIFICATION)
    return benchmark


def create_core50_exp_dataset(imglistpath, root, bp, val_perc=0):
    targets = []
    images = []
    with open(imglistpath, 'r') as f:
        lines = f.readlines()
    for l in lines:
        imgpath, class_id = l.strip().split(' ')
        with Image.open(str(root / bp / imgpath)) as m:
            images.append(m.copy())
        targets.append(int(class_id))
        assert 0 <= targets[-1] <= 49

    tds = CustomExpDataTargets(images, targets)
    dataset = AvalancheDataset(
        dataset=tds,
        task_labels=0,
        dataset_type=AvalancheDatasetType.CLASSIFICATION,
        transform=None)

    if val_perc > 0:
        train_idx, test_idx = train_test_split(range(len(targets)), test_size=val_perc, stratify=targets, shuffle=True)
        val_dataset = AvalancheSubset(dataset, test_idx)
        dataset = AvalancheSubset(dataset, train_idx)

    if val_perc > 0:
        return dataset, val_dataset
    else:
        return dataset


class CustomRobertaClassificationHead(torch.nn.Module):
    """No pooling layer"""
    def __init__(self, num_labels, hidden_size=768):
        super().__init__()
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.out_proj(x)
        return x


def get_filter_indices(values, k, mode='top'):
    """
    mode: 'top': selects indices of top-k values, 'bottom' selects indices of smallest-k values,
        'median' selects k indices around median value
    """
    if mode == 'median':
        sorted_indices = np.argsort(values)
        median_index = len(sorted_indices) // 2
        start = median_index - (k // 2)
        end = median_index + (k // 2)
        indices = sorted_indices[start:end].tolist()
    elif mode == 'top':
        indices = torch.topk(torch.tensor(values), k=k).indices.tolist()
    elif mode == 'bottom':
        indices = torch.topk(-torch.tensor(values), k=k).indices.tolist()
    return indices

@torch.no_grad()
def select_informative_examples(trd, model, device, n_samples=2000, mode='top'):
    """Given a training set from Huggingface datasets, compute the loss
    of the model on each example from the training set. Returns a new Huggingface
    dataset containing examples selected via the mode parameter based on the loss values.
    """

    if mode == 'random':
        rand_indices = list(range(len(trd)))
        random.shuffle(rand_indices)
        rand_indices = rand_indices[:n_samples]
        return trd.filter(lambda el, i: i in rand_indices, with_indices=True)

    losses = []
    for i in tqdm(range(len(trd))):
        example = trd[i]
        input_ids = example['input_ids'].unsqueeze(0).to(device)
        attention_mask = example['attention_mask'].unsqueeze(0).to(device)
        labels = example['input_ids'].unsqueeze(0).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.cpu().item()
        losses.append(loss)
    indices = get_filter_indices(losses, n_samples, mode=mode)

    return trd.filter(lambda el, i: i in indices, with_indices=True)

