import globalvars
import config
import argparse
import framenet_loader

import torch

from datetime import datetime

from torch.utils.data import DataLoader

import models

import utils
from models import CandidateTargetClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import globalvars
import config
import argparse
import framenet_loader

import torch

from datetime import datetime

from torch.utils.data import DataLoader

import models

import utils
from transformers import BertConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import spacy
import xml.etree.ElementTree as ET
import re

import nltk
from nltk.stem import SnowballStemmer
import inflect
import target_utils
from torch.utils.data import WeightedRandomSampler


def main(args):
    print("Loading LU parser...")
    lu_manager = target_utils.LexicalUnitManager()
    lu_manager.load_lus()
    
    # Load datasets
    print("Loading dataset...")
    _train_target, _test_targets = target_utils.load_target_dataset(filter=1)
    
    _train_dataset = target_utils.get_cached_target_dataset(_train_target, "train", lu_manager)
    _test_dataset = target_utils.get_cached_target_dataset(_test_targets, "test", lu_manager)

    train_dataset = target_utils.TargetSpanDataset(_train_dataset)
    test_dataset = target_utils.TargetSpanDataset(_test_dataset)

    # Since batch_size = 1, need grad_accumulation=4 or maybe 8
    import IPython
    IPython.embed()
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # Load model
    print("Loading model...")
    model = CandidateTargetClassifier.from_pretrained("bert-base-uncased").to(globalvars.device)
    
    # Train model
    if args.train:
        print("Beginning model training...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.75)
        
        model.train()
        
        best_acc = 0
        steps = 0
        
        for epoch in range(args.epochs):
            model.train()
            
            for batch in train_dataloader:
                toks = batch["tokens"].to(globalvars.device)
                candidates = batch["label"].to(globalvars.device)
                
                outputs = model(toks, labels=candidates)
                outputs.loss.backward()
                
                if steps % args.grad_accumulation == 0: 
                    optimizer.step()
                    optimizer.zero_grad()

                if steps % 10 == 0:
                    print(f"{epoch} | {steps}: {outputs.loss}")
                
                steps += 1

            lr_scheduler.step()
            
            acc = target_utils.evaluate_targets(model, test_dataloader)
            
            if acc > best_acc:
                best_acc = acc
                print(f"New best acc: {best_acc}")
                torch.save(model, f"{args.model_path}targ-id-{args.model_name if args.model_name != None else 'best'}")

        print(f"Training complete. Best accuracy: {best_acc}")
    
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train and evaluate frame element identification model")
    parser.add_argument("--train", help="train the model from scratch", default=False, action="store_true")
    parser.add_argument("--evaluate", help="evaluate the model", default=True)
    parser.add_argument("--semafor", help="train and evaluate on the same dataset as semafor and open-sesame", default=True)
    parser.add_argument("--epochs", help="number of epochs to train the model on", default=25, type=int)
    parser.add_argument("--grad_accumulation", help="number of batches to accumulate the gradient on", default=8, type=int)
    parser.add_argument("--eval_path", help="directory to save evals in", default="./evals/", type=str)
    parser.add_argument("--model_path", help="dir to save model in", default="./models/")
    parser.add_argument("--eval_after_each_epoch", help="flag for evaluating after each epoch", default=False, action="store_true")
    parser.add_argument("--lr_scheduler", help="which lr scheduler to use", default=None, type=str)
    parser.add_argument("--use_example_sentences", help="use example sentences in frame files", default=False, action="store_true")
    parser.add_argument("--shuffle_dataset", help="shuffle dataset, can help distribute samples better", default=False, action="store_true")
    parser.add_argument("--emb_agg_method", help="method used to aggregate embeddings of the same partition", default="mean", type=str) # mean, first, sum, dim-max
    parser.add_argument("--device", help="device to use when training (use this setting if training with multiple GPUs)", default="cpu", type=str) # usually cuda:0, cuda:1, etc.
    parser.add_argument("--model_name", help="name for the model to be saved as, otherwise will be saved as current time", default=None, type=str) # usually cuda:0, cuda:1, etc.
    
    parser.add_argument("--use_pos_tags", help="", default=False, action="store_true")
    parser.add_argument("--use_lu_sentences", help="", default=False, action="store_true")
    
    args = parser.parse_args()
    
    if args.device:
        globalvars.device = torch.device(args.device)

    main(args)
