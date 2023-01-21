import globalvars
import torch

import sklearn.metrics
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_separate_fes(eval_preds, eval_labels, eval_frames):
    fe_preds = {fr:[] for fr in globalvars.fe_encoders} # frame -> fe predictions for frame
    fe_true = {fr:[] for fr in globalvars.fe_encoders} # frame -> fe gt for frame

    for pred, frame, label in zip(eval_preds, eval_frames, eval_labels):
        fe_preds[frame].append(pred[1:-1]) # ignore padding toks @ 0 and -1
        fe_true[frame].append(label[label != -100]) # ignore padding toks @ 0 and -1

    fe_preds = {fr:torch.cat(x).cpu() for fr, x in fe_preds.items() if len(x) > 0}
    fe_true = {fr:torch.cat(x).cpu() for fr, x in fe_true.items() if len(x) > 0}
    
    return fe_preds, fe_true

def get_separate_fes_no_partitions(eval_preds, eval_labels, eval_frames):
    fe_preds = {fr:[] for fr in globalvars.fe_encoders} # frame -> fe predictions for frame
    fe_true = {fr:[] for fr in globalvars.fe_encoders} # frame -> fe gt for frame

    eval_preds = torch.tensor(pad_sequences(eval_preds, maxlen=globalvars.max_token_len, padding="post", value=-100))
    
    for pred, frame, label in zip(eval_preds, eval_frames, eval_labels):
        fe_preds[frame].append(pred[label != -100]) # ignore padding toks @ 0 and -1
        fe_true[frame].append(label[label != -100]) # ignore padding toks @ 0 and -1

    fe_preds = {fr:torch.cat(x).cpu() for fr, x in fe_preds.items() if len(x) > 0}
    fe_true = {fr:torch.cat(x).cpu() for fr, x in fe_true.items() if len(x) > 0}
    
    return fe_preds, fe_true

def evaluate_fes(model, test_dataloader):
    model.eval()
    eval_preds = []
    eval_labels = []
    eval_frames = []
    
    for batch_toks, batch_frames, batch_fes, batch_partitions in test_dataloader:
        
        outputs = model(input_ids=batch_toks.to(globalvars.device), 
                        partitions=batch_partitions.to(globalvars.device), frames=batch_frames)
        
        eval_preds += [x.argmax(dim=1).detach().cpu() for x in outputs.logits]
        eval_labels += batch_fes
        eval_frames += batch_frames
        
    return eval_preds, eval_labels, eval_frames

def get_fe_metrics(fe_preds, fe_gts):
    fe_metrics = []

    for fr in fe_preds:
        pr = fe_preds[fr]
        gt = fe_gts[fr]
        
        if len(gt) == 0:
            continue

        micro_metrics = sklearn.metrics.precision_recall_fscore_support(gt, pr, labels=np.unique(gt), zero_division=0, average="micro")
        macro_metrics = sklearn.metrics.precision_recall_fscore_support(gt, pr, labels=np.unique(gt), zero_division=0, average="macro")
        weighted_metrics = sklearn.metrics.precision_recall_fscore_support(gt, pr, labels=np.unique(gt), zero_division=0, average="weighted")

        fe_metrics.append({"frame":fr, "micro_precision":micro_metrics[0], "micro_recall":micro_metrics[1], "micro_f1":micro_metrics[2],
                            "macro_precision":macro_metrics[0], "macro_recall":macro_metrics[1], "macro_f1":macro_metrics[2],
                            "weighted_precision":weighted_metrics[0], "weighted_recall":weighted_metrics[1], "weighted_f1":weighted_metrics[2]})
        
    return pd.DataFrame(fe_metrics)

def make_none_partitions(fe_labels):
    # Average Macro f1:               0.404513922042773
    # Average Micro f1:               0.565661293114997
    # Average Weighted f1:            0.59651419188765

    # Average Macro recall:           0.5593291687149193
    # Average Micro recall:           0.5370875236407827
    # Average Weighted recall:        0.5370875236407827

    # Average Macro precision:        0.43024774218088524
    # Average Micro precision:        0.6070958938606099
    # Average Weighted precision:     0.7711646936188717

    # Average Accuracy:               0.5370875236407827
    partitions = torch.arange(fe_labels.shape[1]).repeat(fe_labels.shape[0], 1)
    for i in range(fe_labels.shape[0]):
        final_pos = (fe_labels[i] == -100).nonzero()[1][0]
        partitions[i, final_pos:] = partitions[i, final_pos] 
        
    return partitions

def make_basic_partitions(fe_labels, input_ids):
    # Average Macro f1:               0.4102215257901594                                                          
    # Average Micro f1:               0.5788114058217398
    # Average Weighted f1:            0.608633801454944

    # Average Macro recall:           0.5645153317462527
    # Average Micro recall:           0.5498558989795194
    # Average Weighted recall:        0.5498558989795194

    # Average Macro precision:        0.43120133293218205
    # Average Micro precision:        0.6205106504314427
    # Average Weighted precision:     0.7719567037311404

    # Average Accuracy:               0.5498558989795194
    partitions = []
    
    for i in range(fe_labels.shape[0]):
        part = [0]
        for tok in globalvars.tokenizer.convert_ids_to_tokens(input_ids[i])[1:]:
            if tok.startswith("##"):
                part.append(part[-1])
            else:
                part.append(part[-1] + 1)
        partitions.append(part)
        
    partitions = pad_sequences(partitions, padding="post", value=-100, maxlen=globalvars.max_partition_toks)
    return torch.tensor(partitions)

def evaluate_fes_with_defs(model, test_dataloader):
    model.eval()
    eval_preds = {}
        
    for batch_toks, batch_fes, batch_tok_types, batch_sents, batch_frames in test_dataloader:
        outputs = model(input_ids=batch_toks.to(globalvars.device), token_type_ids=batch_tok_types.to(globalvars.device))
        preds = outputs.logits.detach().cpu()
        
        for i in range(len(batch_sents)):
            if batch_sents[i] not in eval_preds:
                eval_preds[batch_sents[i]] = {}
            
            if batch_frames[i] not in eval_preds[batch_sents[i]]:
                eval_preds[batch_sents[i]][batch_frames[i]] = []
            
            eval_preds[batch_sents[i]][batch_frames[i]].append(preds[i])
    
    eval_preds = {sent:{fr:torch.stack(preds).squeeze().argmax(dim=0) for fr, preds in fr_preds.items()} for sent,fr_preds in eval_preds.items()}

    return eval_preds

def simple_evaluate_fes_with_defs(model, test_dataloader):
    model.eval()
    eval_preds = []
    eval_labels = []
    eval_frames = []
    
    prev_toks = None
    prev_frame = None
    
    sent_preds = []
    sent_labels = []
    
    for batch_toks, batch_fes, batch_tok_types, batch_sents, batch_frames in test_dataloader:
        outputs = model(input_ids=batch_toks.to(globalvars.device), token_type_ids=batch_tok_types.to(globalvars.device))
        batch_preds = outputs.logits.detach().cpu()
        
        if prev_toks == None:
            prev_toks = batch_toks[0]
            prev_frame = batch_frames[0]

        for toks, fes, preds, frame in zip(batch_toks, batch_fes, batch_preds, batch_frames):
            first_sep = (toks == globalvars.tokenizer.sep_token_id).nonzero()[0][0]
            if (toks[:first_sep] == prev_toks[:first_sep]).all():
                sent_preds.append(preds[:first_sep+1])
                sent_labels.append(fes[:first_sep+1])
            else:
                if len(eval_preds) % 1000 == 0:
                    print(f"Evaluating model [{len(eval_preds)}/{len(test_dataloader)}]")
                eval_preds.append(torch.stack(sent_preds).argmax(dim=-1).argmax(dim=0))
                eval_labels.append(torch.stack(sent_labels).argmax(dim=0))
                eval_frames.append(prev_frame)

                sent_preds = [preds[:first_sep+1]]
                sent_labels = [fes[:first_sep+1]]
                
            prev_toks = toks
            prev_frame = frame
        
    return eval_preds, eval_labels, eval_frames

def get_separate_fes_from_defs(eval_preds, eval_labels, eval_frames):
    fe_preds = {fr:[] for fr in globalvars.fe_encoders} # frame -> fe predictions for frame
    fe_true = {fr:[] for fr in globalvars.fe_encoders} # frame -> fe gt for frame

    for pred, frame, label in zip(eval_preds, eval_frames, eval_labels):
        fe_preds[frame].append(pred[1:-1]) # ignore padding toks @ 0 and -1
        fe_true[frame].append(label[1:-1]) # ignore padding toks @ 0 and -1

    fe_preds = {fr:torch.cat(x).cpu() for fr, x in fe_preds.items() if len(x) > 0}
    fe_true = {fr:torch.cat(x).cpu() for fr, x in fe_true.items() if len(x) > 0}
    
    return fe_preds, fe_true

def unpartition_labels(toks, partitions, labels):
    unpartitioned_labels = []
    # unpartition tokens
    for ids, parts, _labels in zip(toks, partitions, labels):
        
        # Make new label by repeating label
        unpart_label = [_labels[parts[i]] for i in range(len(parts))]
        new_label = []
        
        # merge token labels back to word labels
        _toks = globalvars.tokenizer.convert_ids_to_tokens(ids)
        for i in range(1, len(_toks)+1):
            if unpart_label[i] == -100:
                break
            if _toks[i].startswith("##"):
                continue
            new_label.append(unpart_label[i])
        
        unpartitioned_labels.append(torch.tensor(new_label))
        
    return unpartitioned_labels

def make_simple_partitions(input_ids):

    partitions = []
    
    part = [0]
    for i, tok in enumerate(globalvars.tokenizer.convert_ids_to_tokens(input_ids)[1:]):
        if tok == "[PAD]":
            break
        if tok.startswith("##"):
            part.append(part[-1])
        else:
            part.append(part[-1] + 1)
    partitions.append(part)
        
    partitions = pad_sequences(partitions, padding="post", value=part[-1]+1, maxlen=globalvars.max_partition_toks)
    return torch.tensor(partitions)

def eval_targets(model, test_dataloader):
    model.eval()
    eval_preds = []
    eval_labels = []
    
    for batch_toks, batch_partitions, batch_targets in test_dataloader:
        outputs = model(input_ids=batch_toks.to(globalvars.device), partitions=batch_partitions.to(globalvars.device), 
                        labels=batch_targets)
        preds = [x.argmax(dim=1).detach().cpu() for x in outputs.logits]
        eval_preds += preds
        eval_labels += [batch_targets[i][:preds[i].shape[0]] for i in range(len(preds))]
        del outputs, preds
    return eval_preds, eval_labels

def get_target_metrics(eval_preds, eval_labels):
   
    micro_metrics = sklearn.metrics.precision_recall_fscore_support(eval_labels, eval_preds, zero_division=0, average="micro")
    macro_metrics = sklearn.metrics.precision_recall_fscore_support(eval_labels, eval_preds, zero_division=0, average="macro")
    weighted_metrics = sklearn.metrics.precision_recall_fscore_support(eval_labels, eval_preds, zero_division=0, average="weighted")
    metrics = {"micro_precision":micro_metrics[0], "micro_recall":micro_metrics[1], "micro_f1":micro_metrics[2],
            "macro_precision":macro_metrics[0], "macro_recall":macro_metrics[1], "macro_f1":macro_metrics[2],
            "weighted_precision":weighted_metrics[0], "weighted_recall":weighted_metrics[1], "weighted_f1":weighted_metrics[2]}
    return pd.DataFrame([metrics])
