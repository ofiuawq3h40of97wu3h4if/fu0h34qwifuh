import argparse
import os
import pandas as pd
import seaborn as sns

import config

import matplotlib.pyplot as plt

def sample_analysis(args):
    import numpy as np
    
    epoch_metrics = []
    
    files_in_dir = [f for f in os.listdir(args.eval_path) if f.endswith("-eval.csv") and args.eval_model in f]
    assert len(files_in_dir) > 0, "Couldn't find eval files in specified --eval_path"
    
    num_total_epochs = int(files_in_dir[0].split("-")[2])

    # Assuming these are saved by fe_identification.py, using format fe-id-25-1-2022-10-15-21-07-eval.csv
    for i in range(num_total_epochs):
        if not os.path.isfile(f"{args.eval_path}fe-id-{num_total_epochs}-{i+1}-{args.eval_model}-eval.csv"):
            break
        
        df = pd.read_csv(f"{args.eval_path}fe-id-{num_total_epochs}-{i+1}-{args.eval_model}-eval.csv")
        df["epoch"] = i+1
        
        epoch_metrics.append(df)
    
    full_df = pd.concat(epoch_metrics)
    
    frame_counts = np.load("./data/train_frame_counts.npy", allow_pickle=True)
    
    # Plot 
    sns.set_theme()
    
    x_vals = []
    y_vals = []
    
    for i in range(30):
        temp_df = full_df.copy()
        for filtered_frame in [fr for fr, count in frame_counts if count > i]:
            temp_df = temp_df[temp_df.frame != filtered_frame]
        max_vals = temp_df.groupby('epoch').mean().max().round(decimals=3)
        
        x_vals.append(i)
        y_vals.append(max_vals.macro_f1)
    
    sns.lineplot(x=x_vals, y=y_vals).set(title=f"Model: {args.eval_model}", xlabel="Frame samples in training set", ylabel="Macro F1")
    
    plt.show()


def eval_epochs(args):
    epoch_metrics = []
    
    files_in_dir = [f for f in os.listdir(args.eval_path) if f.endswith("-eval.csv") and args.eval_model in f]
    assert len(files_in_dir) > 0, "Couldn't find eval files in specified --eval_path"
    
    num_total_epochs = int(files_in_dir[0].split("-")[2])

    # Assuming these are saved by fe_identification.py, using format fe-id-25-1-2022-10-15-21-07-eval.csv
    if args.only_epoch == None:
        for i in range(num_total_epochs):
            if not os.path.isfile(f"{args.eval_path}fe-id-{num_total_epochs}-{i+1}-{args.eval_model}-eval.csv"):
                break
            
            df = pd.read_csv(f"{args.eval_path}fe-id-{num_total_epochs}-{i+1}-{args.eval_model}-eval.csv")
            df["epoch"] = i+1
            
            epoch_metrics.append(df)
        
        full_df = pd.concat(epoch_metrics)
    
    else:
        if not os.path.isfile(f"{args.eval_path}fe-id-{num_total_epochs}-{args.only_epoch}-{args.eval_model}-eval.csv"):
            assert False, "Unable to find specified epoch file."
        
        df = pd.read_csv(f"{args.eval_path}fe-id-{num_total_epochs}-{args.only_epoch}-{args.eval_model}-eval.csv")
        df["epoch"] = args.only_epoch
        
        epoch_metrics.append(df)
        
        full_df = pd.concat(epoch_metrics)

    if args.remove_k_samples != None:
        import numpy as np
        frame_counts = np.load("./data/train_frame_counts.npy", allow_pickle=True)
        
        for filtered_frame in [fr for fr, count in frame_counts if count > args.remove_k_samples]:
            full_df = full_df[full_df.frame != filtered_frame]

    # Plot 
    sns.set_theme()
    
    max_vals = full_df.groupby('epoch').mean().max().round(decimals=3)
    max_epoch = full_df.groupby('epoch').mean().idxmax()
    
    print(f"Best Macro f1:              {max_vals.macro_f1} @ {max_epoch.macro_f1} epochs")
    print(f"Best Micro f1:              {max_vals.micro_f1} @ {max_epoch.micro_f1} epochs")
    print(f"Best Weighted f1:           {max_vals.weighted_f1} @ {max_epoch.weighted_f1} epochs")
    print("")
    print(f"Best Macro recall:          {max_vals.macro_recall} @ {max_epoch.macro_recall} epochs")
    print(f"Best Micro recall:          {max_vals.micro_recall} @ {max_epoch.micro_recall} epochs")
    print(f"Best Weighted recall:       {max_vals.weighted_recall} @ {max_epoch.weighted_recall} epochs")
    print("")
    print(f"Best Macro precision:       {max_vals.macro_precision} @ {max_epoch.macro_precision} epochs")
    print(f"Best Micro precision:       {max_vals.micro_precision} @ {max_epoch.micro_precision} epochs")
    print(f"Best Weighted precision:    {max_vals.weighted_precision} @ {max_epoch.weighted_precision} epochs")
    print("")
    print(f"Best Accuracy:              {max_vals.micro_recall} @ {max_epoch.micro_recall} epochs")
    
    plot_metrics = ["macro_f1", "micro_f1", "weighted_f1", "micro_recall"]
    for metric in plot_metrics:
        sns.lineplot(full_df.groupby("epoch").mean(), x="epoch", y=metric, label=metric).set(title=f"Model: {args.eval_model}")
    plt.show()

def full_model_eval(args):
    import framenet_loader
    import numpy as np
    import torch
    import globalvars
    import config
    import pickle
    from torch.utils.data import DataLoader
    import models
    import utils
    from sklearn.metrics import accuracy_score
    
    print("Loading metadata")
    # Get framenet metadata
    frame_info = framenet_loader.get_frame_info(f"{config.framenet_path}/frame/")
    frame_encoder = framenet_loader.LabelEncoder().fit([x["name"] for x in frame_info])
    fe_encoders = {frame["name"]:framenet_loader.LabelEncoder().fit(frame["fes"]) for frame in frame_info}
    
    # Set global variables
    globalvars.frame_encoder = frame_encoder
    globalvars.fe_encoders = fe_encoders
    
    print("Loading evaluation set")
    # load evaluation set
    test_dataset_files = [f"{config.framenet_path}/fulltext/{x}" for x in config.OPENSESAME_TEST_FILES]
    data_loader = framenet_loader.FrameNetDataLoader(globalvars.tokenizer, fe_encoders, frame_encoder)
    if os.path.isfile(f"{config.cache_dir}test_dataset.pickle"):
        with open(f"{config.cache_dir}test_dataset.pickle", "rb") as f:
            test_dataset = pickle.load(f)
    else:
        test_dataset = data_loader.get_fulltext_data(test_dataset_files)
        with open(f"{config.cache_dir}test_dataset.pickle", "wb") as f:
            pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)
    
    test_toks, test_fes, test_targets, test_frames, test_sents = data_loader.pad_datasets(test_dataset)
    
    test_toks = torch.tensor(test_toks)
    test_fes = torch.tensor(test_fes)
    test_targets = torch.tensor(test_targets)
    test_frames = np.array(test_frames)
    test_sents = np.array(test_sents)
    
    if args.partition_method == "ground_truth":
        test_partitions = torch.vstack(framenet_loader.make_partitions(test_fes))
        
    elif args.partition_method == "basic":
        globalvars.max_partition_toks = 160
        globalvars.max_token_len = 160 if globalvars.max_token_len == None else globalvars.max_token_len
        test_partitions = utils.make_basic_partitions(test_fes, test_toks, globalvars.tokenizer)
        test_partitions = torch.vstack([torch.tensor(x) for x in test_partitions])
        
    elif args.partition_method == "none":
        globalvars.max_partition_toks = 160
        globalvars.max_token_len = 160 if globalvars.max_token_len == None else globalvars.max_token_len
        test_partitions = utils.make_none_partitions(test_fes)
        
    test_partitioned_fes = framenet_loader.partition_labels(test_partitions, test_fes)
    test_dataloader = DataLoader(list(zip(test_toks, test_frames, test_partitioned_fes, test_partitions)), batch_size=args.batch_size)
    
    if args.cache_preds:
        if os.path.isfile(f"{config.cache_dir}{args.eval_model}-preds.pickle"):
            print("Loading cached preds")
            with open(f"{config.cache_dir}{args.eval_model}-preds.pickle", "rb") as f:
                eval_preds, eval_labels, eval_frames = pickle.load(f)
        else:
            print("Evaluating model")
            model = torch.load(f"{args.model_path}{args.eval_model}")
            eval_preds, eval_labels, eval_frames = utils.evaluate_fes(model, test_dataloader)
            print("Caching preds")
            with open(f"{config.cache_dir}{args.eval_model}-preds.pickle", "wb") as f:
                pickle.dump((eval_preds, eval_labels, eval_frames), f, pickle.HIGHEST_PROTOCOL)
    else:
        model = torch.load(f"{args.model_path}{args.eval_model}")
        eval_preds, eval_labels, eval_frames = utils.evaluate_fes(model, test_dataloader)
    
    
    if args.partition_method in ["none", "basic"]:
        fe_preds, fe_gts = utils.get_separate_fes_no_partitions(eval_preds, eval_labels, eval_frames)
    elif args.partition_method == "unpartition":
        fe_gts = utils.unpartition_labels(test_toks, test_partitions, test_partitioned_fes)
        fe_preds = utils.unpartition_labels(test_toks, test_partitions, eval_preds)
    else:
        fe_preds, fe_gts = utils.get_separate_fes(eval_preds, eval_labels, eval_frames)
    
    fe_metrics = utils.get_fe_metrics(fe_preds, fe_gts)
    
    print(f"Mean Frame Macro f1:               {fe_metrics.macro_f1.mean().round(decimals=3)}")
    print(f"Mean Frame Micro f1:               {fe_metrics.micro_f1.mean().round(decimals=3)}")
    print(f"Mean Frame Weighted f1:            {fe_metrics.weighted_f1.mean().round(decimals=3)}")
    print("")
    print(f"Mean Frame Macro recall:           {fe_metrics.macro_recall.mean().round(decimals=3)}")
    print(f"Mean Frame Micro recall:           {fe_metrics.micro_recall.mean().round(decimals=3)}")
    print(f"Mean Frame Weighted recall:        {fe_metrics.weighted_recall.mean().round(decimals=3)}")
    print("")
    print(f"Mean Frame Macro precision:        {fe_metrics.macro_precision.mean().round(decimals=3)}")
    print(f"Mean Frame Micro precision:        {fe_metrics.micro_precision.mean().round(decimals=3)}")
    print(f"Mean Frame Weighted precision:     {fe_metrics.weighted_precision.mean().round(decimals=3)}")
    print("")
    print(f"Mean Frame Accuracy:               {fe_metrics.micro_recall.mean().round(decimals=3)}")
    print(f"Overall Accuracy:               {round(accuracy_score(torch.cat(eval_labels)[torch.cat(eval_labels) != -100], torch.cat([x[1:-1] for x in eval_preds])), 3)}")
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    
    plot_row = 0
    plot_col = 0
    for col in fe_metrics.columns[1:]:
        g = sns.histplot(x=fe_metrics[col], bins=np.arange(0, 1.1, 0.1), ax=axs[plot_row][plot_col])

        if plot_col == 2:
            plot_row += 1
            plot_col = 0
        else:
            plot_col += 1
    
    fig.savefig(f"{args.output_name}.png", format="png", dpi=500)

def all_model_eval(args):
    name_mapping = {"2022-10-19-05-24":"Aggregation = first", 
                    "2022-10-19-11-37":"Aggregation = mean", 
                    "2022-10-19-09-41":"Aggregation = sum", 
                    "2022-10-19-07-46":"Aggregation = max", 
                    "2022-10-18-17-31":"Using examples (step 0.5/5)", 
                    "2022-10-21-15-59":"Using examples (adam 1e-3 + step)", 
                    "2022-10-16-21-48":"step test",
                    "2022-10-16-15-11":"exp test",
                    "2022-10-16-08-29":"linear test",
                    }
    
    epoch_metrics = []
    
    files_in_dir = [f for f in os.listdir(args.eval_path) if f.endswith("-eval.csv")]
    assert len(files_in_dir) > 0, "Couldn't find eval files in specified --eval_path"
    
    seen_models = set([])
    
    max_num_epochs = 25
    
    for eval_file in files_in_dir:
        eval_model = "-".join(eval_file.split("-")[4:-1])
        if eval_model in seen_models:
            continue
    
        num_total_epochs = int(eval_file.split("-")[2])

        # Assuming these are saved by fe_identification.py, using format fe-id-25-1-2022-10-15-21-07-eval.csv
        for i in range(max_num_epochs):
            if os.path.isfile(f"{args.eval_path}fe-id-{num_total_epochs}-{i+1}-{eval_model}-eval.csv"):
                df = pd.read_csv(f"{args.eval_path}fe-id-{num_total_epochs}-{i+1}-{eval_model}-eval.csv")
                df["epoch"] = i+1
                df["model"] = name_mapping.get(eval_model, eval_model)
            
            epoch_metrics.append(df) # keep appending last epoch's results until end

        seen_models.add(eval_model)
        
    full_df = pd.concat(epoch_metrics)
    
    if args.remove_out_of_domain:
        for ood_fr in config.out_of_domain:
            full_df = full_df[full_df.frame != ood_fr]
    
    # Plot 
    # sns.set_theme()
    sns.set_palette("bright")
    fig, ax = plt.subplots(1,1)
    g = sns.lineplot(full_df.groupby(["model", "epoch"]).mean(), x="epoch", y="macro_f1", hue="model")
    fig.suptitle("Full model comparison")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Macro F1", fontsize=14)
    # plt.tight_layout()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"{args.output_name}.png", format="png", dpi=500, bbox_inches='tight')

def __process_conll(conll):
    pred_sents = []
    pred_fes = []
    pred_frames = []
    
    cur_frame = None
    cur_words = []
    cur_fes = []
    
    for line in conll:
        tok_num = int(line[0])
        word = line[3]
        fe = "[NONFETOKEN]" if line[-1] == "O" else line[-1][2:]
        
        if tok_num == 1 and len(cur_words) > 0:
            assert cur_frame != None
            pred_sents.append(" ".join(cur_words))
            pred_fes.append(cur_fes)
            pred_frames.append(cur_frame)
            
            cur_frame = None
            cur_words = []
            cur_fes = []

        if line[-2] != "_":
            cur_frame = line[-2]
        
        cur_words.append(word)
        cur_fes.append(fe)
    
    pred_sents.append(" ".join(cur_words))
    pred_fes.append(cur_fes)
    pred_frames.append(cur_frame)
    
    return pred_sents, pred_fes, pred_frames

def eval_conll(args):
    import numpy as np
    import framenet_loader
    import utils
    import seaborn
    import pandas as pd
    import sklearn.metrics
    
    conll_preds = np.loadtxt(f"{args.pred_conll}", delimiter="\t", dtype=str) # "./other/predicted-1.7-argid-test.conll"
    conll_gts = np.loadtxt(f"{args.gt_conll}", delimiter="\t", dtype=str) # "./other/predicted-1.7-argid-test.conll"
    
    pred_sents, pred_fes, pred_frames = __process_conll(conll_preds)
    gt_sents, gt_fes, gt_frames = __process_conll(conll_gts)
    
    frame_info = framenet_loader.get_frame_info(f"{config.framenet_path}/frame/")
    frame_encoder = framenet_loader.LabelEncoder().fit([x["name"] for x in frame_info])
    fe_encoders = {frame["name"]:framenet_loader.LabelEncoder().fit(frame["fes"]) for frame in frame_info}

    pred_enc_fes = []
    for sent, frame, fes in zip(pred_sents, pred_frames, pred_fes):
        pred_enc_fes.append(fe_encoders[frame].transform(fes))
    
    gt_enc_fes = []
    for sent, frame, fes in zip(gt_sents, gt_frames, gt_fes):
        gt_enc_fes.append(fe_encoders[frame].transform(fes))
    
    fe_preds = {fr:[] for fr in fe_encoders}
    fe_gts = {fr:[] for fr in fe_encoders}
    
    for pred, gt, frame in zip(pred_enc_fes, gt_enc_fes, gt_frames):
        fe_preds[frame].append(pred)
        fe_gts[frame].append(gt)
    
    fe_preds = {fr:np.concatenate(x) for fr, x in fe_preds.items() if len(x) > 0}
    fe_gts = {fr:np.concatenate(x) for fr, x in fe_gts.items() if len(x) > 0}
    
    fe_metrics = utils.get_fe_metrics(fe_preds, fe_gts)
    

    print(f"Test sentences:                     {len(pred_sents)}")
    print(f"Test frames:                        {len(pred_frames)}")
    print(f"Test FEs:                           {len(pred_fes)}")
    
    print(f"Mean Frame Macro f1:                {fe_metrics.macro_f1.mean().round(decimals=3)}")
    print(f"Mean Frame Micro f1:                {fe_metrics.micro_f1.mean().round(decimals=3)}")
    print(f"Mean Frame Weighted f1:             {fe_metrics.weighted_f1.mean().round(decimals=3)}")
    print("")
    print(f"Mean Frame Macro recall:            {fe_metrics.macro_recall.mean().round(decimals=3)}")
    print(f"Mean Frame Micro recall:            {fe_metrics.micro_recall.mean().round(decimals=3)}")
    print(f"Mean Frame Weighted recall:         {fe_metrics.weighted_recall.mean().round(decimals=3)}")
    print("")
    print(f"Mean Frame Macro precision:         {fe_metrics.macro_precision.mean().round(decimals=3)}")
    print(f"Mean Frame Micro precision:         {fe_metrics.micro_precision.mean().round(decimals=3)}")
    print(f"Mean Frame Weighted precision:      {fe_metrics.weighted_precision.mean().round(decimals=3)}")
    print("")
    print(f"Mean Frame Accuracy:                {fe_metrics.micro_recall.mean().round(decimals=3)}")
    
    print(f"Overall Accuracy:                   {round(sklearn.metrics.accuracy_score(np.concatenate(gt_enc_fes), np.concatenate(pred_enc_fes)), 3)}")
    print(f"Overall F1-Score:                   {round(sklearn.metrics.f1_score(np.concatenate(gt_enc_fes), np.concatenate(pred_enc_fes), average='micro'), 3)}")
    print(f"Overall Precision:                  {round(sklearn.metrics.precision_score(np.concatenate(gt_enc_fes), np.concatenate(pred_enc_fes), average='micro'), 3)}")
    print(f"Overall Recall:                     {round(sklearn.metrics.recall_score(np.concatenate(gt_enc_fes), np.concatenate(pred_enc_fes), average='micro'), 3)}")

def eval_dataset(args):
    import framenet_loader
    import numpy as np
    from sklearn.metrics import accuracy_score
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import Counter
    import globalvars
    
    sns.set_style("white")
    
    # Load dataset
    (train_toks, train_fes, train_targets, train_frames, 
    train_sents), (test_toks, test_fes, test_targets, 
    test_frames, test_sents), model_config = framenet_loader.load_dataset(args)
        
    # import IPython
    # IPython.embed()
    
    # Print number of samples
    print(f"\nTraining set:")
    print(f"\tSentences:               {np.unique(train_sents).shape[0]}")
    print(f"\tFrames:                  {np.unique(train_frames).shape[0]}")
    print(f"\tSentence-frame pairs:    {train_frames.shape[0]}")
    print(f"\tSentence-FE pairs:       {sum([x.unique().shape[0] > 1 for x in train_fes])}")
    
    print(f"\nEvaluation set:")
    print(f"\tSentences:               {np.unique(test_sents).shape[0]}")
    print(f"\tFrames:                  {np.unique(test_frames).shape[0]}")
    print(f"\tSentence-frame pairs:    {test_frames.shape[0]}")
    print(f"\tSentence-FE pairs:       {sum([x.unique().shape[0] > 1 for x in test_fes])}")
   
    train_frame_counter = Counter(train_frames)
    test_frame_counter = Counter(test_frames)
    
    real_counts = list(train_frame_counter.values()) + list([0] * (globalvars.frame_encoder.classes_.shape[0] - len(train_frame_counter)))
    
    print(f"\t% frames with <= 5:        {len([x for x in real_counts if x < 5]) / len(real_counts)}")
    print(f"\t% frames with <= 10:       {len([x for x in real_counts if x < 10]) / len(real_counts)}")
    print(f"\t# frames with <= 5:        {len([x for x in real_counts if x < 5])}")
    print(f"\t# frames with <= 10:       {len([x for x in real_counts if x < 10])}")
    
    
    # frame distribution plot
    fig, axs = plt.subplots(1,1, sharex=True, sharey=True, figsize=(5,4))

    tr_plt = sns.histplot(real_counts, ax=axs, binrange=(0, 50), binwidth=5, color="#7EA6E0", edgecolor="#558AD6", alpha=0.5)

    axs.set_ylabel("Number of frames")
    axs.set_xlabel("Number of sentences")
    axs.set_ylim(0, 801)
    axs.set_title("Training set frame distribution")
    # axs.get_legend().remove()
    plt.tight_layout()
    fig.savefig(f"frame-distribution.png", format="png", dpi=1000)
    
    frame_sample_counts = [0]
    
    for x in sorted(real_counts, reverse=True):
        frame_sample_counts.append(frame_sample_counts[-1] + (100 * x / sum(train_frame_counter.values())))
    
    # frame sample % plot
    fig, axs = plt.subplots(1,1, sharex=True, sharey=True, figsize=(5,4))
    
    tr_plt = sns.lineplot(frame_sample_counts, ax=axs, color="#558AD6")

    axs.set_ylabel("% of training sentences")
    axs.set_xlabel("N most common frames")
    axs.set_ylim(0, 101)
    axs.set_title("Training set frame distribution")
    axs.fill_between(range(len(frame_sample_counts)), frame_sample_counts, color="#7EA6E0", alpha=0.5)
    plt.tight_layout()
    fig.savefig(f"frame-distribution-line.png", format="png", dpi=1000)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Evaluate frame element identification model prediction/metrics")
    parser.add_argument("--eval_path", help="directory to model evaluations are saved in", default="./evals/", type=str)
    parser.add_argument("--eval_epochs", help="evaluate performance of model over each epoch", default=False, action="store_true")
    parser.add_argument("--eval_model", help="model to evaluate, for --eval_epochs, this is the datetime string, e.g. 2022-10-15-21-07", 
                        default=None, type=str)
    parser.add_argument("--eval_frame", help="frame to evaluate", default=None, type=str)
    parser.add_argument("--output_name", help="frame to evaluate", default="untitled", type=str)
    
    parser.add_argument("--full_eval", help="full evaluation of model", default=False, action="store_true")
    parser.add_argument("--model_path", help="directory of model (for full eval)", default="./models/", type=str)
    parser.add_argument("--batch_size", help="", default=8, type=int)
    parser.add_argument("--cache_preds", help="", default=False, action="store_true")
    parser.add_argument("--partition_method", help="which partition method to use during evaluation", 
                        default="ground_truth", choices=["ground_truth", "basic", "none"])
    
    parser.add_argument("--all_evals", help="", default=False, action="store_true")
    parser.add_argument("--remove_k_samples", help="", default=None, type=int)
    parser.add_argument("--sample_analysis", help="", default=False, action="store_true")
    
    parser.add_argument("--eval_conll", help="", default=False, action="store_true")
    parser.add_argument("--pred_conll", help="", type=str, default=None)
    parser.add_argument("--gt_conll", help="", type=str, default=None)
    
    parser.add_argument("--eval_dataset", help="", default=False, action="store_true")
    parser.add_argument("--use_examples", help="", default=False, action="store_true")
    parser.add_argument("--semafor", help="", type=str, default=True)
    parser.add_argument("--train", help="", type=str, default=True)
    parser.add_argument("--evaluate", help="", type=str, default=True)
    
    
    parser.add_argument("--only_epoch", help="", type=int, default=None)
    
    args = parser.parse_args()
    
    if args.eval_epochs:
        eval_epochs(args)
    elif args.full_eval:
        assert args.eval_model != None, "Enter model name using --eval_model model_name"
        full_model_eval(args)
    elif args.all_evals:
        all_model_eval(args)
    elif args.sample_analysis:
        sample_analysis(args)
    elif args.eval_conll:
        assert args.pred_conll != None, "Please specify --pred_conll"
        assert args.gt_conll != None, "Please specify --gt_conll"
        eval_conll(args)
    elif args.eval_dataset:
        eval_dataset(args)
        
        