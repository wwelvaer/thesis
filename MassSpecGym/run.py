import argparse
import datetime
import typing as T
from pathlib import Path

from rdkit import RDLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os.path
import pickle

import numpy as np
print(np.version.version)

import massspecgym.utils as utils
from massspecgym.data import RetrievalDataset, MassSpecDataset, MassSpecDataModule
from massspecgym.data.transforms import (
    MolFingerprinter, SpecBinner, SpecTokenizer, MolToFormulaVector
)
from massspecgym.models.base import Stage
from massspecgym.models.retrieval import (
    FingerprintFFNRetrieval, FromDictRetrieval, RandomRetrieval, DeepSetsRetrieval
)
#from massspecgym.models.de_novo import SmilesTransformer
from smiles_transformer import SmilesTransformer
from massspecgym.models.tokenizers import SmilesBPETokenizer, SelfiesTokenizer
from massspecgym.definitions import MASSSPECGYM_TEST_RESULTS_DIR


# Suppress RDKit warnings and errors
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# TODO Organize configs better (probably with hydra)
parser = argparse.ArgumentParser()

# Submission
parser.add_argument('--job_key', type=str, required=True)

# Experiment setup
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--project_name', type=str, default=None)
parser.add_argument('--wandb_entity_name', type=str, default='mass-spec-ml')
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test_only', action='store_true')

# Data paths
parser.add_argument('--candidates_pth', type=str, default=None)
parser.add_argument('--dataset_pth', type=str, default=None,
    help='Path to the dataset file in the .tsv or .mgf format.')
parser.add_argument('--split_pth', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=1)

# Data transforms setup

# - Binner
parser.add_argument('--max_mz', type=int, default=1005)
parser.add_argument('--bin_width', type=float, default=1)

# - Tokenizer
parser.add_argument('--n_peaks', type=int, default=60)

# - Fingerprinter
parser.add_argument('--fp_size', type=int, default=4096)

# Training setup
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--accelerator', type=str, default='gpu')
parser.add_argument('--devices', type=int, default=1)
parser.add_argument('--log_every_n_steps', type=int, default=50)
parser.add_argument('--val_check_interval', type=float, default=1.0)
parser.add_argument('--check_val_every_n_epoch', type=int, default=1)

# General hyperparameters
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', type=float, default=0.0)

# Task and model
parser.add_argument('--task', type=str, choices=['retrieval', 'de_novo', 'simulation'], required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--log_only_loss_at_stages', default=(),
    type=lambda stages: [Stage(s) for s in stages.strip().replace(' ', '').split(',')])
parser.add_argument('--df_test_pth', type=Path, default=None)
parser.add_argument('--checkpoint_pth', type=Path, default=None)
parser.add_argument('--no_checkpoint', type=bool, default=False)

# - De novo

# 1. SmilesTransformer
parser.add_argument('--input_dim', type=int, default=2)
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--num_encoder_layers', type=int, default=3)
parser.add_argument('--num_decoder_layers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--k_predictions', type=int, default=10)
parser.add_argument('--pre_norm', type=bool, default=False)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--smiles_tokenizer', default='tokenizers/smiles_tokenizer_4M.pkl')
parser.add_argument('--full_selfies_vocab', action='store_true')
parser.add_argument('--use_chemical_formula', action='store_true')

parser.add_argument('--sampler', type=str, default='naive')
parser.add_argument('--k', type=int, default=50)
parser.add_argument('--mz_scaling', type=bool, default=False)
parser.add_argument('--patience', type=int, default=0)
parser.add_argument('--embedding_norm', type=bool, default=False)
parser.add_argument('--q', type=float, default='0.8')
parser.add_argument('--beam_width', type=int, default='20')
parser.add_argument('--alpha', type=float, default='1.0')
parser.add_argument('--store_metadata', action='store_true')

def main(args):
    print(args)
    # Seed everything
    pl.seed_everything(args.seed)

    # Get current time
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Process args
    if args.df_test_pth is None and args.devices == 1:
        args.df_test_pth = MASSSPECGYM_TEST_RESULTS_DIR / f"{args.task}/{args.run_name}_{now_formatted}.pkl"

    # Init paths to data files
    if args.debug:
        args.dataset_pth = "../data/debug/example_5_spectra.mgf"
        args.candidates_pth = "../data/debug/example_5_spectra_candidates.json"
        args.split_pth="../data/debug/example_5_spectra_split.tsv"

    # Load dataset
    if args.task == 'retrieval':
        if args.model == 'fingerprint_ffn':
            spec_transform = SpecBinner(max_mz=args.max_mz, bin_width=args.bin_width)
        else:
            spec_transform = SpecTokenizer(n_peaks=args.n_peaks, matchms_kwargs=dict(mz_to=args.max_mz))
        dataset = RetrievalDataset(
            pth=args.dataset_pth,
            spec_transform=spec_transform,
            mol_transform=MolFingerprinter(fp_size=args.fp_size),
            candidates_pth=args.candidates_pth,
        )
    elif args.task == 'de_novo':
        dataset = MassSpecDataset(
            pth=args.dataset_pth,
            spec_transform = SpecTokenizer(n_peaks=args.n_peaks, matchms_kwargs=dict(mz_to=args.max_mz)),
            mol_transform={'formula': MolToFormulaVector(), 'mol': None} if args.use_chemical_formula else None
        )
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")

    # Init data module
    data_module = MassSpecDataModule(
        dataset=dataset,
        split_pth=args.split_pth,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Init model
    common_kwargs = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_only_loss_at_stages=args.log_only_loss_at_stages,
        df_test_path=args.df_test_pth,
    )
    if args.task == 'retrieval':
        if args.model == 'fingerprint_ffn':
            model = FingerprintFFNRetrieval(
                in_channels=int(args.max_mz * (1 / args.bin_width)),
                hidden_channels=args.hidden_channels,
                out_channels=args.fp_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                **common_kwargs
            )
        elif args.model == 'deepsets':
            model = DeepSetsRetrieval(
                in_channels=2,
                hidden_channels=args.hidden_channels,
                out_channels=args.fp_size,
                num_layers_per_mlp=args.num_layers_per_mlp,
                dropout=args.dropout,
                **common_kwargs
            )
        elif args.model == 'from_dict':
            model = FromDictRetrieval(
                dct_path=args.dct_path,
                **common_kwargs
            )
        elif args.model == 'random':
            model = RandomRetrieval(
                **common_kwargs
            )
        else:
            raise NotImplementedError(f"Model {args.model} not implemented.")
    elif args.task == 'de_novo':
        if args.model == 'smiles_transformer':
            if not os.path.isfile(args.smiles_tokenizer):
                raise FileNotFoundError(f"Tokenizer {args.smiles_tokenizer} not found")
            else:
                with open(args.smiles_tokenizer, 'rb') as file:
                    smiles_tokenizer = pickle.load(file)

            model = SmilesTransformer(
                input_dim=args.input_dim,
                d_model=args.d_model,
                nhead=args.nhead,
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers,
                dropout=args.dropout,
                smiles_tokenizer=smiles_tokenizer,
                k_predictions=args.k_predictions,
                pre_norm=args.pre_norm,
                max_smiles_len=smiles_tokenizer.max_length,
                chemical_formula=args.use_chemical_formula,
                sampler=args.sampler,
                k=args.k,
                q=args.q,
                mz_scaling=args.mz_scaling,
                embedding_norm=args.embedding_norm,
                beam_width = args.beam_width,
                alpha = args.alpha,
                store_metadata = args.store_metadata,
                **common_kwargs
            )
        else:
            raise NotImplementedError(f"Model {args.model} not implemented.")
    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")

    # If checkpoint path is provided, load the model from the checkpoint instead
    # and override the parameters not related to the model architecture and training
    # TODO Extend to pass arguments to be overridden as an argument to the script
    # For example: --override_args="df_test_path,lr,hidden_channels"
    if args.checkpoint_pth is not None:
        model = type(model).load_from_checkpoint(
            args.checkpoint_pth,
            log_only_loss_at_stages=args.log_only_loss_at_stages,
            df_test_path=args.df_test_pth
        )
        model.sampler = args.sampler
        model.k_predictions = args.k_predictions
        print(type(model), model.sampler)
        model.k = args.k
        model.q = args.q
        model.temperature = args.temperature
        model.beam_width = args.beam_width
        model.alpha = args.alpha
        print(args.store_metadata)
        model.store_metadata = args.store_metadata

    # Init logger
    if args.no_wandb:
        logger = None
    else:
        logger = pl.loggers.WandbLogger(
            name=args.run_name,
            project=args.project_name,
            log_model=False,
            config=args
        )

    # Init callbacks for checkpointing and early stopping
    callbacks = []
    for i, monitor in enumerate(model.get_checkpoint_monitors()):
        monitor_name = monitor['monitor']
        if not args.no_checkpoint:
            checkpoint = pl.callbacks.ModelCheckpoint(
                monitor=monitor_name,
                save_top_k=1,
                mode=monitor['mode'],
                dirpath=Path(args.project_name) / args.run_name,
                filename=f'{{step:06d}}-{{{monitor_name}:03.03f}}',
                auto_insert_metric_name=True,
                save_last=(i == 0)
            )
            callbacks.append(checkpoint)
        if args.patience > 0 and monitor.get('early_stopping', False):
            early_stopping = EarlyStopping(
                monitor=monitor_name,
                mode=monitor['mode'],
                verbose=True,
                patience=args.patience
            )
            callbacks.append(early_stopping)

    # Init trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        #check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        enable_checkpointing=not args.no_checkpoint,
    )

    # Prepare data module to validate or test before training
    data_module.prepare_data()
    data_module.setup()

    if not args.test_only:
        # Validate before training
        trainer.validate(model, datamodule=data_module)

        # Train
        trainer.fit(model, datamodule=data_module)
    
    model.log_only_loss_at_stages = [Stage("train")]
    trainer.validate(model, datamodule=data_module)
    model.log_only_loss_at_stages = args.log_only_loss_at_stages

    # Test
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.project_name is None:
        task_name = args.task.replace('_', ' ').title().replace(' ', '')
        args.project_name = f"MassSpecGym{task_name}"

    main(args)
