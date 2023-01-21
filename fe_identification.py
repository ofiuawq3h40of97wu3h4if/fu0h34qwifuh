import globalvars
import config
import argparse
import framenet_loader

import torch

from itertools import chain
from datetime import datetime

from torch.utils.data import DataLoader

import models

import utils
import os
import numpy as np

class DummyArgs():
    def __init__(self):
        self.train = False
        self.evaluate = True
        self.semafor = True
        self.epochs = 25
        self.batch_size = 8
        self.eval_path = "./models/"
        self.model_path = None
        self.cpu = False
        self.eval_after_each_epoch = False
        self.lr_scheduler = None
        self.cyclic_step_epochs = 2
        self.use_example_sentences = False
        self.shuffle_dataset = False
        self.emb_agg_method = "mean"

def main(args):
    print(f"Using device: {globalvars.device}")
    
    print("Loading FrameNet fulltext annotations...")
    
    # Load framenet
    if args.use_deberta:
        (train_toks, train_fes, train_targets, train_frames, 
        train_sents), (test_toks, test_fes, test_targets, 
        test_frames, test_sents), model_config = framenet_loader.load_dataset_for_deberta(args)
    else:
        (train_toks, train_fes, train_targets, train_frames, 
        train_sents), (test_toks, test_fes, test_targets, 
        test_frames, test_sents), model_config = framenet_loader.load_dataset(args)
    
    # Create partitions for FEs
    if args.partition_method == "gt":
        train_partitions = torch.vstack(framenet_loader.make_partitions(train_fes))
        test_partitions = torch.vstack(framenet_loader.make_partitions(test_fes))
    elif args.partition_method == "basic":
        globalvars.max_partition_toks = 168
        train_partitions = utils.make_basic_partitions(train_fes, train_toks)
        test_partitions = utils.make_basic_partitions(test_fes, test_toks)
    elif args.partition_method == "none":
        globalvars.max_partition_toks = 168
        train_partitions = utils.make_none_partitions(train_fes, train_toks)
        test_partitions = utils.make_none_partitions(test_fes, test_toks)

    # Partition FEs
    train_partitioned_fes = framenet_loader.partition_labels(train_partitions, train_fes)
    test_partitioned_fes = framenet_loader.partition_labels(test_partitions, test_fes)

    if args.use_example_sentences:
        print("Loading FrameNet example sentences...")
        example_toks, example_frames, example_fes = framenet_loader.load_example_sentences(args)
        example_partitions = torch.vstack(framenet_loader.make_partitions(example_fes))
        example_partitioned_fes = framenet_loader.partition_labels(example_partitions, example_fes)

        train_toks = torch.cat([train_toks, example_toks])
        train_partitions = torch.cat([train_partitions, example_partitions])
        train_partitioned_fes = np.concatenate([train_partitioned_fes, example_partitioned_fes])
        train_frames = np.concatenate([train_frames, example_frames])
    
    if args.use_lu_sentences:
        pass

    train_dataloader = DataLoader(list(zip(train_toks, train_frames, train_partitioned_fes, train_partitions)), batch_size=args.batch_size, shuffle=args.shuffle_dataset)
    test_dataloader = DataLoader(list(zip(test_toks, test_frames, test_partitioned_fes, test_partitions)), batch_size=args.batch_size)

    # Create model
    # load pretrained BERT model, finetune it
    print("Loading model...")
    if args.use_deberta:
        model = models.DebertaFrameElementIdentifier.from_pretrained("microsoft/deberta-base", config=model_config).to(globalvars.device)
    
    elif args.emb_agg_method == "mean":
        model = models.FrameElementIdentifier.from_pretrained("bert-base-cased", config=model_config).to(globalvars.device)
    elif args.emb_agg_method == "first":
        model = models.FrameElementIdentifierAggFirst.from_pretrained("bert-base-cased", config=model_config).to(globalvars.device)
    elif args.emb_agg_method == "sum":
        model = models.FrameElementIdentifierAggSum.from_pretrained("bert-base-cased", config=model_config).to(globalvars.device)
    elif args.emb_agg_method == "max":
        model = models.FrameElementIdentifierAggFirst.from_pretrained("bert-base-cased", config=model_config).to(globalvars.device)
    
    bert_optim = torch.optim.AdamW(model.bert.parameters(), lr=1e-5)
    fe_optim = torch.optim.AdamW(chain(*[x.parameters() for fr, x in model.fe_classifiers.items()]))
    
    if args.lr_scheduler != None:
        if args.lr_scheduler == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=fe_optim, gamma=0.9) #
        elif args.lr_scheduler == "linear":
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=fe_optim, start_factor=1, end_factor=0.1, total_iters=10)
        elif args.lr_scheduler == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=fe_optim, step_size=5, gamma=0.5)
        elif args.lr_scheduler == "cyclic":
            bert_optim = torch.optim.SGD(model.bert.parameters(), lr=1e-3)
            fe_optim = torch.optim.SGD(chain(*[x.parameters() for fr, x in model.fe_classifiers.items()]), lr=1e-3)
            
            bert_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=fe_optim, base_lr=1e-3, max_lr=2e-1, mode="triangular2", step_size_up= int(args.cyclic_step_epochs * train_toks.shape[0] / args.batch_size))
            fe_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=fe_optim,  base_lr=1e-3, max_lr=2e-1, mode="triangular2", step_size_up= int(args.cyclic_step_epochs * train_toks.shape[0] / args.batch_size))
    
    # train model
    train_start_time = datetime.now()
    model_name = args.model_name if args.model_name != None else train_start_time.strftime('%Y-%m-%d-%H-%M')
    best_macro_f1 = 0 # store best model based on macro f1
    
    print("Beginning model training... ")
    print(f"Model will be saved at {args.eval_path}fe-id-{args.epochs}-{model_name}.")
    for epoch in range(args.epochs):
        model.train()
        steps = 0
        
        for batch_toks, batch_frames, batch_fes, batch_partitions in train_dataloader:
            import IPython
            IPython.embed()
            batch_fes = [batch_fes[i][:batch_partitions[i][-1]+1].type(torch.LongTensor).to(globalvars.device) for i in range(len(batch_fes))]
            
            outputs = model(input_ids=batch_toks.to(globalvars.device), partitions=batch_partitions.to(globalvars.device), 
                            labels=batch_fes, frames=batch_frames)
            
            if steps % 1000 == 0:
                print(f"[{epoch}, {steps}] Loss: {outputs.loss}")
            
            if outputs.loss != None:
                outputs.loss.backward()

            bert_optim.step()
            fe_optim.step()
            
            if args.lr_scheduler in ["cyclic"]:
                bert_scheduler.step()
                fe_scheduler.step()
            
            bert_optim.zero_grad()
            fe_optim.zero_grad()
            
            steps += args.batch_size
        
        if args.lr_scheduler != None and not args.lr_scheduler in ["cyclic"]:
            lr_scheduler.step()
        
        if args.eval_after_each_epoch:
            model.eval()
            
            eval_preds, eval_labels, eval_frames = utils.evaluate_fes(model, test_dataloader)
            if args.partition_method in ["basic", "none"]:
                fe_preds, fe_gts = utils.get_separate_fes_no_partitions(eval_preds, eval_labels, eval_frames)
            else:
                fe_preds, fe_gts = utils.get_separate_fes(eval_preds, eval_labels, eval_frames)
            fe_metrics = utils.get_fe_metrics(fe_preds, fe_gts)

            fe_metrics.to_csv(f"{args.eval_path}fe-id-{args.epochs}-{epoch+1}-{model_name}-eval.csv")
            
            if fe_metrics.macro_f1.mean() > best_macro_f1:
                best_macro_f1 = fe_metrics.macro_f1.mean()
                
                print(f"Saving best model: Macro F1 = {best_macro_f1}")
                torch.save(model, f"{args.model_path}fe-id-{args.epochs}-{model_name}-best")
                print(f"Model saved to: {args.model_path}fe-id-{args.epochs}-{model_name}")
            
            del eval_preds, eval_labels, eval_frames, fe_preds, fe_gts, fe_metrics

    if args.evaluate:
        print("Evaluating model...")
        
        eval_preds, eval_labels, eval_frames = utils.evaluate_fes(model, test_dataloader)
        fe_preds, fe_gts = utils.get_separate_fes(eval_preds, eval_labels, eval_frames)
        fe_metrics = utils.get_fe_metrics(fe_preds, fe_gts)
        
        now = datetime.now()
        fe_metrics.to_csv(f"{args.eval_path}fe-id-{args.epochs}-{now.strftime('%Y-%m-%d-%H-%M')}-eval.csv")
        
        print(f"Average Macro f1:               {fe_metrics.macro_f1.mean()}")
        print(f"Average Micro f1:               {fe_metrics.micro_f1.mean()}")
        print(f"Average Weighted f1:            {fe_metrics.weighted_f1.mean()}")
        print("")
        print(f"Average Macro recall:           {fe_metrics.macro_recall.mean()}")
        print(f"Average Micro recall:           {fe_metrics.micro_recall.mean()}")
        print(f"Average Weighted recall:        {fe_metrics.weighted_recall.mean()}")
        print("")
        print(f"Average Macro precision:        {fe_metrics.macro_precision.mean()}")
        print(f"Average Micro precision:        {fe_metrics.micro_precision.mean()}")
        print(f"Average Weighted precision:     {fe_metrics.weighted_precision.mean()}")
        print("")
        print(f"Average Accuracy:               {fe_metrics.micro_recall.mean()}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train and evaluate frame element identification model")
    parser.add_argument("--train", help="train the model from scratch", default=False, action="store_true")
    parser.add_argument("--evaluate", help="evaluate the model", default=True)
    parser.add_argument("--semafor", help="train and evaluate on the same dataset as semafor and open-sesame", default=True)
    parser.add_argument("--epochs", help="number of epochs to train the model on", default=25, type=int)
    parser.add_argument("--batch_size", help="number of samples in each batch", default=8, type=int)
    parser.add_argument("--eval_path", help="directory to save evals in", default="./evals/", type=str)
    parser.add_argument("--model_path", help="dir to save model in", default="./models/")
    parser.add_argument("--cpu", help="flag for using the cpu instead of cuda", default=False, action="store_true")
    parser.add_argument("--eval_after_each_epoch", help="flag for evaluating after each epoch", default=False, action="store_true")
    parser.add_argument("--lr_scheduler", help="which lr scheduler to use", default=None, type=str)
    parser.add_argument("--cyclic_step_epochs", help="number of epochs for a single cycle", default=2, type=int)
    parser.add_argument("--use_example_sentences", help="use example sentences in frame files", default=False, action="store_true")
    parser.add_argument("--shuffle_dataset", help="shuffle dataset, can help distribute samples better", default=False, action="store_true")
    parser.add_argument("--emb_agg_method", help="method used to aggregate embeddings of the same partition", default="mean", type=str) # mean, first, sum, dim-max
    parser.add_argument("--device", help="device to use when training (use this setting if training with multiple GPUs)", default=None, type=str) # usually cuda:0, cuda:1, etc.
    parser.add_argument("--model_name", help="name for the model to be saved as, otherwise will be saved as current time", default=None, type=str) # usually cuda:0, cuda:1, etc.
    parser.add_argument("--multi_gpu", help="use multiple GPUs if available (not available right now)", default=False, action="store_true")
    parser.add_argument("--partition_method", help="", default="gt", choices=["gt", "basic", "none"]) 
    
    parser.add_argument("--use_deberta", help="", default=False, action="store_true")
    parser.add_argument("--use_pos_tags", help="", default=False, action="store_true")
    parser.add_argument("--use_lu_sentences", help="", default=False, action="store_true")
    
    args = parser.parse_args()
    
    if args.cpu:
        globalvars.device = torch.device("cpu")
    
    if args.device:
        globalvars.device = torch.device(args.device)

    main(args)

# export TF_CPP_MIN_LOG_LEVEL=3 to remove tf error