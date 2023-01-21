import os
from statistics import mode
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
from transformers import PretrainedConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import numpy as np

import globalvars
import config

import pickle

class FrameNetDataLoader():
    def __init__(self, tokenizer, fe_encoders, frame_encoder):
        self.tokenizer = tokenizer
        self.fe_encoders = fe_encoders
        self.frame_encoder = frame_encoder
    
    def pad_datasets(self, dataset, max_len=160):
        padded_fes = pad_sequences([x["fes"] if x["fes"] is not None else [-100] for x in dataset], maxlen=max_len, padding="post", value=-100)
        padded_sentence_tokens = pad_sequences([x["sentence_tokens"] if x["sentence_tokens"] is not None else [] for x in dataset], maxlen=max_len, padding="post", value=0)
        padded_targets = pad_sequences([x["targets"] if x["targets"] is not None else [] for x in dataset], maxlen=max_len, padding="post", value=-100)
        padded_frame = [x["frame"] for x in dataset]
        padded_sentence = [x["sentence"] for x in dataset]

        return padded_sentence_tokens, padded_fes, padded_targets, padded_frame, padded_sentence

    def get_sentences(self, node):
        return node.findall(".//{http://framenet.icsi.berkeley.edu}sentence")
    
    def get_annotations(self, node):
        return node.findall(".//{http://framenet.icsi.berkeley.edu}annotationSet[@status='MANUAL']")
    
    def get_text(self, node):
        return node.find("{http://framenet.icsi.berkeley.edu}text").text

    def get_annotation_targets(self, node):
        return node.findall(".//{http://framenet.icsi.berkeley.edu}label[@name='Target']")

    def get_annotation_fes(self, node):
        return node.findall(".//{http://framenet.icsi.berkeley.edu}layer[@name='FE'][@rank='1']")

    def get_annotation_target_spans(self, annotation_targets, frame):
        target_spans = []
        if len(annotation_targets) > 0:
            for target in annotation_targets:
                target_spans.append((int(target.attrib["start"]), int(target.attrib["end"]), frame))
        return target_spans

    def get_annotation_fe_spans(self, annotation_fes):
        fe_spans = []
        for _fes in annotation_fes:

            for fe in _fes:
                start = int(fe.attrib.get('start', '-1'))
                end = int(fe.attrib.get('end', '-1'))
                name = fe.attrib.get('name')

                if start >= 0:
                    fe_spans.append((start, end, name))

        return sorted(fe_spans)

    def get_sent_partitions(self, text, spans):
        partitions = []

        last = 0
        for span in spans:
            if last == span[0] or text[last:span[0]].strip() == '':
                # handle situation that would cause: ('', '')
                partitions.append((text[span[0]:span[1] + 1].strip(), span[2]))

            else:
                # general situation
                partitions.append((text[last:span[0]].strip(), '')) 
                partitions.append((text[span[0]:span[1] + 1].strip(), span[2])) 

            if span == spans[-1] and span[1] + 1 < len(text):
                # append the rest after processing the last span
                partitions.append((text[span[1] + 1:].strip(), ''))

            last = span[1] + 1
        
        return partitions

    def get_target_label(self, sent, target_spans):
        targ_label = [-100]
        for part, t in self.get_sent_partitions(sent, target_spans):
            if t == '':
                _label = -100
            else:
                _label = self.frame_encoder.transform([t])[0]

            enc_part = self.tokenizer.encode(part)[1:-1]
            targ_label += [_label] * len(enc_part)

        targ_label += [-100]
        return targ_label

    def get_simple_target_label(self, sent, target_spans):
        targ_label = [-100]
        for part, t in self.get_sent_partitions(sent, target_spans):
            if t == '':
                _label = 0
            else:
                _label = 1

            enc_part = self.tokenizer.encode(part)[1:-1]
            targ_label += [_label] * len(enc_part)

        targ_label += [-100]
        return targ_label

    def get_fe_label(self, sent, fe_spans, frame):
        fe_labels = [-100]

        for part, label in self.get_sent_partitions(sent, fe_spans):
            if label == '':
                _label = self.fe_encoders[frame].transform(["[NONFETOKEN]"])[0] # class n = non fe token
            else:
                _label = self.fe_encoders[frame].transform([label])[0]

            enc_part = self.tokenizer.encode(part)[1:-1]
            fe_labels += [_label] * len(enc_part)

        fe_labels += [-100]
        
        assert fe_labels[0] == -100, "Label creation error"
        
        return fe_labels

    def get_fulltext_data(self, data_files):
        # Fulltext data
        dataset = []

        # Loop through each file in dataset
        for fi in data_files:
            root = ET.parse(fi).getroot()
            sentences = self.get_sentences(root)

            for i, sent in enumerate(sentences):
                annos = self.get_annotations(sent)
                sent_text = self.get_text(sent)
                tokenized_sent = self.tokenizer.encode(sent_text)
                assert tokenized_sent[0] == self.tokenizer.cls_token_id and tokenized_sent[-1] == self.tokenizer.sep_token_id, "Tokenization error"

                if len(annos) == 0:
                    continue # ignore unannotated sentences

                # Each annotation
                for anno in annos:
                    frame_name = anno.attrib.get('frameName')
                    
                    if frame_name == None or frame_name == "":
                        continue

                    anno_targets = self.get_annotation_targets(anno)
                    anno_fes = self.get_annotation_fes(anno)

                    target_spans = sorted(list(set(self.get_annotation_target_spans(anno_targets, frame_name))))
                    fe_spans = self.get_annotation_fe_spans(anno_fes)

                    target_label = self.get_target_label(sent_text, target_spans)

                    fe_label = self.get_fe_label(sent_text, fe_spans, frame_name)

                    dataset.append({"sentence":sent_text, "sentence_tokens":tokenized_sent, "frame":frame_name, "fes":fe_label if len(fe_spans) > 0 else None, "targets":target_label if len(target_spans) > 0 else None})

        return dataset

    def get_fulltext_data_with_pos(self, data_files):
        # Fulltext data
        dataset = []

        # Loop through each file in dataset
        for fi in data_files:
            root = ET.parse(fi).getroot()
            sentences = self.get_sentences(root)

            for i, sent in enumerate(sentences):
                annos = self.get_annotations(sent)
                sent_text = self.get_text(sent)
                tokenized_sent = self.tokenizer.encode(sent_text)
                assert tokenized_sent[0] == self.tokenizer.cls_token_id and tokenized_sent[-1] == self.tokenizer.sep_token_id, "Tokenization error"

                if len(annos) == 0:
                    continue # ignore unannotated sentences

                # Each annotation
                for anno in annos:
                    frame_name = anno.attrib.get('frameName')
                    
                    if frame_name == None or frame_name == "":
                        continue

                    anno_targets = self.get_annotation_targets(anno)
                    anno_fes = self.get_annotation_fes(anno)

                    target_spans = sorted(list(set(self.get_annotation_target_spans(anno_targets, frame_name))))
                    fe_spans = self.get_annotation_fe_spans(anno_fes)

                    target_label = self.get_target_label(sent_text, target_spans)

                    fe_label = self.get_fe_label(sent_text, fe_spans, frame_name)

                    dataset.append({"sentence":sent_text, "sentence_tokens":tokenized_sent, "frame":frame_name, "fes":fe_label if len(fe_spans) > 0 else None, "targets":target_label if len(target_spans) > 0 else None})

        return dataset

    def get_all_targets(self):
        import sys
        # Fulltext data
        dataset = []

        # Loop through each file in dataset
        for fi in os.listdir(f"{config.framenet_path}/fulltext/"): # do the same with lu dir and maybe frame dir
            root = ET.parse(f"{config.framenet_path}/fulltext/{fi}").getroot()
            sentences = self.get_sentences(root)

            for i, sent in enumerate(sentences):
                annos = self.get_annotations(sent)
                sent_text = self.get_text(sent)
                tokenized_sent = self.tokenizer.encode(sent_text)
                assert tokenized_sent[0] == self.tokenizer.cls_token_id and tokenized_sent[-1] == self.tokenizer.sep_token_id, "Tokenization error"

                if len(annos) == 0:
                    continue # ignore unannotated sentences

                # Each annotation
                for anno in annos:
                    frame_name = anno.attrib.get('frameName')
                    
                    if frame_name == None or frame_name == "":
                        continue

                    anno_targets = self.get_annotation_targets(anno)
                    target_spans = sorted(list(set(self.get_annotation_target_spans(anno_targets, frame_name))))
                    target_label = self.get_simple_target_label(sent_text, target_spans)
                    dataset.append({"sentence":sent_text, "sentence_tokens":tokenized_sent, "target":target_label})

        return dataset

class FrameBertConfig(PretrainedConfig):
    # mostly copied from huggingface BertConfig
    model_type = "bert"

    def __init__(self, supported_frames=None, all_fes=None, fe_encoders=None, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, 
        hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, 
        type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False, 
        position_embedding_type="absolute", use_cache=True, classifier_dropout=0.1, **kwargs):
        
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.supported_frames = supported_frames
        
        self.all_fes = all_fes
        self.fe_encoders = fe_encoders

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

def get_frame_info(path):
    # Gets frame information extracted from FN dataset
    # Sorts data by frame name before returning
    # Returns Name, FEs, and #FEs for each frame

    frame_info = []
    for x in os.listdir(path):
        if x.endswith(".xml"):
            frame_root = ET.parse(path+x).getroot()
            fes = [x.attrib["name"] for x in frame_root.findall("{http://framenet.icsi.berkeley.edu}FE")]
            frame_info.append({"name":frame_root.attrib["name"], "fes":["[NONFETOKEN]"] + fes, "num_elements":len(fes)+1})
    
    return sorted(frame_info, key=lambda x: x["name"])

def load_dataset(args):
    # Get frame information
    frame_info = get_frame_info(f"{config.framenet_path}/frame/")

    # Create frame label encoder to convert frame name -> label
    frame_encoder = LabelEncoder().fit([x["name"] for x in frame_info])

    # Create FE label encoder to convert FE name -> label, for each frame
    fe_encoders = {frame["name"]:LabelEncoder().fit(frame["fes"]) for frame in frame_info}

    # Create config and pass the frame information to model
    model_config = FrameBertConfig.from_pretrained("bert-base-cased", supported_frames=frame_info)
    
    # Set global variables
    globalvars.frame_encoder = frame_encoder
    globalvars.fe_encoders = fe_encoders
    globalvars.model_config = model_config
    
    # Define dataset files
    if args.semafor:
        train_dataset_files = [f"{config.framenet_path}/fulltext/{x}" for x in config.OPENSESAME_TRAIN_FILES]
        test_dataset_files = [f"{config.framenet_path}/fulltext/{x}" for x in config.OPENSESAME_TEST_FILES]
    else:
        print("This is currently unsupported.")
        return None
    
    # Create data loader
    data_loader = FrameNetDataLoader(globalvars.tokenizer, fe_encoders, frame_encoder)

    # Parse fulltext data
    if args.evaluate and args.train:
        if os.path.isfile(f"{config.cache_dir}train_dataset.pickle"):
            with open(f"{config.cache_dir}train_dataset.pickle", "rb") as f:
                train_dataset = pickle.load(f)
        else:
            train_dataset = data_loader.get_fulltext_data(train_dataset_files)
            with open(f"{config.cache_dir}train_dataset.pickle", "wb") as f:
                pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
            
        if os.path.isfile(f"{config.cache_dir}test_dataset.pickle"):
            with open(f"{config.cache_dir}test_dataset.pickle", "rb") as f:
                test_dataset = pickle.load(f)
        else:
            test_dataset = data_loader.get_fulltext_data(test_dataset_files)
            with open(f"{config.cache_dir}test_dataset.pickle", "wb") as f:
                pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)

    elif args.evaluate:
        if os.path.isfile(f"{config.cache_dir}test_dataset.pickle"):
            with open(f"{config.cache_dir}test_dataset.pickle", "rb") as f:
                test_dataset = pickle.load(f)
        else:
            test_dataset = data_loader.get_fulltext_data(test_dataset_files)
            with open(f"{config.cache_dir}test_dataset.pickle", "wb") as f:
                pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)
            
    else:
        assert False, "--train without --evaluate is currently unsupported"
    
    if args.train:
        train_toks, train_fes, train_targets, train_frames, train_sents = data_loader.pad_datasets(train_dataset, max_len=globalvars.max_token_len)
        
        train_toks = torch.tensor(train_toks)
        train_fes = torch.tensor(train_fes)
        train_targets = torch.tensor(train_targets)
        train_frames = np.array(train_frames)
        train_sents = np.array(train_sents)

    
    test_toks, test_fes, test_targets, test_frames, test_sents = data_loader.pad_datasets(test_dataset, max_len=globalvars.max_token_len)
    
    test_toks = torch.tensor(test_toks)
    test_fes = torch.tensor(test_fes)
    test_targets = torch.tensor(test_targets)
    test_frames = np.array(test_frames)
    test_sents = np.array(test_sents)
    
    if args.train:
        return ((train_toks, train_fes, train_targets, train_frames, train_sents), (test_toks, test_fes, test_targets, test_frames, test_sents), model_config)
    elif args.evaluate:
        return (test_toks, test_fes, test_targets, test_frames, test_sents)

def make_partitions(fe_labels):
    # Creates partitions for fe labels based on ground-truth labels
    partitioned_labels = []

    for fe_lab in fe_labels:
        cur_label = [0]
        part_number = 0

        for fe_i in range(1, len(fe_lab)):
            if fe_lab[fe_i] != fe_lab[fe_i-1]:
                part_number += 1
            cur_label.append(part_number)

        partitioned_labels.append(torch.tensor(cur_label))
    return partitioned_labels

def partition_labels(partitions, labels):
    return pad_sequences([labels[i][partitions[i].diff(prepend=torch.tensor([-1])) != 0].type(torch.long) 
                          for i in range(partitions.shape[0])], padding="post", value=-100, maxlen=globalvars.max_partition_toks)

def load_example_sentences(args):
    # Get frame information
    frame_info = get_frame_info(f"{config.framenet_path}/frame/")

    # Create frame label encoder to convert frame name -> label
    frame_encoder = LabelEncoder().fit([x["name"] for x in frame_info])

    # Create FE label encoder to convert FE name -> label, for each frame
    fe_encoders = {frame["name"]:LabelEncoder().fit(frame["fes"]) for frame in frame_info}
    
    # when loading, need to keep partitions
    # when training, only train for positive samples, i.e. labels should be -100 or a FE
    
    frame_info = []
    full_fe_examples = {}
    
    # Load examples
    for x in os.listdir(f"{config.framenet_path}/frame/"):
        if x.endswith(".xml") and x not in set(["Time_vector.xml"]): # skip time_vector, its bad apparently?
            with open(f"{config.framenet_path}/frame/{x}", 'r') as file:
                xml_string = file.read().replace('&gt;', '>').replace('&lt;', '<') 
            
            frame_root = ET.fromstring(xml_string)
            
            fes = frame_root.findall(".//{http://framenet.icsi.berkeley.edu}FE")
            fe_abbrev = {}
            for fe in fes:
                if "abbrev" in fe.attrib and fe.attrib["abbrev"] not in [None, ""]:
                    fe_abbrev[fe.attrib["abbrev"].lower()] = fe.attrib["name"]
            
            examples = frame_root.findall(".//{http://framenet.icsi.berkeley.edu}ex")
                        
            fe_examples = []
            
            for ex in examples:
                fe_parts = [(ex.text, None)] if ex.text != None and ex.text != ' ' and ex.text != '' else []
                
                for piece in ex.iter():
                    piece_fe = piece.attrib.get("name", None)
                    
                    if piece.text == None or piece.text == "" or piece.text == " ":
                        continue
                    
                    if len(fe_parts) > 0 and piece_fe == fe_parts[-1][1]:
                        fe_parts[-1] = (fe_parts[-1][0] + piece.text, piece_fe)
                    else:
                        if piece_fe != None:
                            fe_parts.append((piece.text, fe_abbrev.get(piece_fe.lower(), piece_fe)))
                        else:
                            fe_parts.append((piece.text, fe_abbrev.get(piece_fe, piece_fe)))
                        
                    if piece == ex[-1] and piece.tail != None:
                        fe_parts.append((piece.tail, None))
                
                fe_examples.append(fe_parts)

            fe_names = [x.attrib["name"] for x in fes]
            frame_info.append({"name":frame_root.attrib["name"], "fes":["[NONFETOKEN]"] + fe_names, "num_elements":len(fe_names)+1})
            full_fe_examples[frame_root.attrib["name"]] = fe_examples
    
    
    fe_example_dataset = []
    # Parse examples into dataset
    for frame, examples in full_fe_examples.items():
        
        for example in examples:
            sent_text = " ".join([x[0].strip() for x in example if x[0] != None])
            sent_toks = globalvars.tokenizer.encode(sent_text)
            
            _fe_label = [-100]
            
            for span_text, span_fe in example:
                span_toks = globalvars.tokenizer.encode(span_text)[1:-1]
                
                if span_fe == None:
                    _fe_label += [-100] * len(span_toks)
                else:
                    if span_fe in fe_encoders[frame].classes_:
                        _fe_label += [fe_encoders[frame].transform([span_fe])[0]] * len(span_toks)
                    else:
                        continue
                    
            _fe_label += [-100]
            
            if len(sent_toks) == len(_fe_label) and len(sent_text) > 30: # basic filter out garbage
                #{"sentence":sent_text, "sentence_tokens":tokenized_sent, "frame":frame_name, 
                # "fes":fe_label if len(fe_spans) > 0 else None}
                fe_example_dataset.append({"sentence":sent_text, "sentence_tokens":sent_toks, 
                                           "frame":frame, "fes":_fe_label})
    
    # Make tensors from dataset
    train_example_fes = pad_sequences([x["fes"] for x in fe_example_dataset], maxlen=globalvars.max_token_len, value=-100, padding="post")
    train_example_toks = pad_sequences([x["sentence_tokens"] for x in fe_example_dataset], maxlen=globalvars.max_token_len, padding="post")
    train_example_frames = np.array([x["frame"] for x in fe_example_dataset])
    
    return (torch.tensor(train_example_toks), train_example_frames, torch.tensor(train_example_fes))

def load_definitions():
    from nltk.tokenize import sent_tokenize
    
    frame_fe_defs = {}
    for x in os.listdir(f"{config.framenet_path}/frame/"):
        if x.endswith(".xml") and x not in set(["Time_vector.xml"]): # skip time_vector, its bad apparently?
            with open(f"{config.framenet_path}/frame/"+x, 'r') as file:
                xml_string = file.read().replace('&gt;', '>').replace('&lt;', '<')
            
            frame_root = ET.fromstring(xml_string)
            frame_name = frame_root.attrib.get("name", None)
            
            frame_def_node = frame_root.find("{http://framenet.icsi.berkeley.edu}definition/{http://framenet.icsi.berkeley.edu}def-root")
            frame_def = sent_tokenize("".join(frame_def_node.itertext()))[0] # Take only first sentence in case theres junk
            
            frame_fe_defs[frame_name] = {"definition":frame_def, "fe_defs":{}}
            
            fes = frame_root.findall(".//{http://framenet.icsi.berkeley.edu}FE")

            for fe in fes:
                fe_name = fe.attrib.get("name", None)
                
                if fe_name == None or fe_name == "":
                    continue
                
                def_node = fe.find(".//{http://framenet.icsi.berkeley.edu}def-root")

                fe_def = "".join(def_node.itertext()) 
                if fe_def == "" or fe_def == None:
                    continue
                
                fe_def = sent_tokenize(fe_def)[0] # same as above, take only first sentences in case of junk
                
                if fe_def.startswith("This FE indicates"):
                    fe_def = fe_def.lstrip("This FE indicates")
                
                frame_fe_defs[frame_name]["fe_defs"][fe_name] = fe_def

    return frame_fe_defs

def get_fe_def_dataset(sents, fe_labels, frames, frame_fe_defs, test_or_train="train"):
    if os.path.isfile(f"{config.cache_dir}frame-def-{test_or_train}.dataset"):
        with open(f"{config.cache_dir}frame-def-{test_or_train}.dataset", "rb") as f:
            new_dataset = pickle.load(f)
        return new_dataset
    
    new_dataset = []
    for sent, fe_label, frame in zip(sents, fe_labels, frames):
        if frame == "Time_vector":
            continue # skip time vector for now
        
        for i in range(globalvars.fe_encoders[frame].classes_.shape[0]):
            cur_fe_label = (fe_label == i) * 1 # set true values equal to 1, others to 0
            cur_fe_label[fe_label == -100] = -100 # keep -100 labels from before
            
            fe_name = globalvars.fe_encoders[frame].inverse_transform([i])[0]
            tokenized_input = globalvars.tokenizer(sent, f'{fe_name} : {frame_fe_defs[frame]["fe_defs"].get(fe_name, "")}')
            new_dataset.append({"sentence_tokens":tokenized_input["input_ids"], "fe_label":cur_fe_label, 
                                "token_type_ids":tokenized_input["token_type_ids"], "sentence":sent, "frame":frame})
    
    with open(f"{config.cache_dir}frame-def-{test_or_train}.dataset", "wb") as f:
        pickle.dump(new_dataset, f, pickle.HIGHEST_PROTOCOL)
    
    return new_dataset

class FrameDebertaConfig(PretrainedConfig):
    model_type = "deberta"

    def __init__(
        self, supported_frames=None, vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, 
        intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, 
        attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=0, 
        initializer_range=0.02, layer_norm_eps=1e-7, relative_attention=False, 
        max_relative_positions=-1, pad_token_id=0, position_biased_input=True, 
        pos_att_type=None, pooler_dropout=0, pooler_hidden_act="gelu",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.supported_frames = supported_frames

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input

        # Backwards compatibility
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act

def load_dataset_for_deberta(args):
    # Get frame information
    frame_info = get_frame_info(f"{config.framenet_path}/frame/")

    # Create frame label encoder to convert frame name -> label
    frame_encoder = LabelEncoder().fit([x["name"] for x in frame_info])

    # Create FE label encoder to convert FE name -> label, for each frame
    fe_encoders = {frame["name"]:LabelEncoder().fit(frame["fes"]) for frame in frame_info}

    # Create config and pass the frame information to model
    model_config = FrameDebertaConfig.from_pretrained("microsoft/deberta-base", supported_frames=frame_info)
    
    from transformers import DebertaTokenizer
    globalvars.tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    
    # Set global variables
    globalvars.frame_encoder = frame_encoder
    globalvars.fe_encoders = fe_encoders
    globalvars.model_config = model_config
    
    # Define dataset files
    if args.semafor:
        train_dataset_files = [f"{config.framenet_path}/fulltext/{x}" for x in config.OPENSESAME_TRAIN_FILES]
        test_dataset_files = [f"{config.framenet_path}/fulltext/{x}" for x in config.OPENSESAME_TEST_FILES]
    else:
        print("This is currently unsupported.")
        return None
    
    # Create data loader
    data_loader = FrameNetDataLoader(globalvars.tokenizer, fe_encoders, frame_encoder)

    # Parse fulltext data
    if args.evaluate and args.train:
        if os.path.isfile(f"{config.cache_dir}train_dataset_deberta.pickle"):
            with open(f"{config.cache_dir}train_dataset_deberta.pickle", "rb") as f:
                train_dataset = pickle.load(f)
        else:
            train_dataset = data_loader.get_fulltext_data(train_dataset_files)
            with open(f"{config.cache_dir}train_dataset_deberta.pickle", "wb") as f:
                pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
            
        if os.path.isfile(f"{config.cache_dir}test_dataset_deberta.pickle"):
            with open(f"{config.cache_dir}test_dataset_deberta.pickle", "rb") as f:
                test_dataset = pickle.load(f)
        else:
            test_dataset = data_loader.get_fulltext_data(test_dataset_files)
            with open(f"{config.cache_dir}test_dataset_deberta.pickle", "wb") as f:
                pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)

    elif args.evaluate:
        if os.path.isfile(f"{config.cache_dir}test_dataset.pickle"):
            with open(f"{config.cache_dir}test_dataset.pickle", "rb") as f:
                test_dataset = pickle.load(f)
        else:
            test_dataset = data_loader.get_fulltext_data(test_dataset_files)
            with open(f"{config.cache_dir}test_dataset.pickle", "wb") as f:
                pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)

    else:
        assert False, "--train without --evaluate is currently unsupported"
    
    if args.train:
        train_toks, train_fes, train_targets, train_frames, train_sents = data_loader.pad_datasets(train_dataset, max_len=globalvars.max_token_len)
        
        train_toks = torch.tensor(train_toks)
        train_fes = torch.tensor(train_fes)
        train_targets = torch.tensor(train_targets)
        train_frames = np.array(train_frames)
        train_sents = np.array(train_sents)

    test_toks, test_fes, test_targets, test_frames, test_sents = data_loader.pad_datasets(test_dataset, max_len=globalvars.max_token_len)
    
    test_toks = torch.tensor(test_toks)
    test_fes = torch.tensor(test_fes)
    test_targets = torch.tensor(test_targets)
    test_frames = np.array(test_frames)
    test_sents = np.array(test_sents)
    
    assert (test_fes[:, 0] == -100).all(), "FEs should start with -100"
    assert train_frames.shape[0] == train_sents.shape[0] == train_targets.shape[0] == train_fes.shape[0] == train_toks.shape[0], "Error loading data"
    assert test_frames.shape[0] == test_sents.shape[0] == test_targets.shape[0] == test_fes.shape[0] == test_toks.shape[0], "Error loading data"
    
    if args.train:
        return ((train_toks, train_fes, train_targets, train_frames, train_sents), (test_toks, test_fes, test_targets, test_frames, test_sents), model_config)
    elif args.evaluate:
        return (test_toks, test_fes, test_targets, test_frames, test_sents)

def load_example_targets(args):
    # Get frame information
    frame_info = get_frame_info(f"{config.framenet_path}/frame/")

    # Create frame label encoder to convert frame name -> label
    frame_encoder = LabelEncoder().fit([x["name"] for x in frame_info])

    frame_info = []
    target_examples = []
    
    # Load examples
    for x in os.listdir(f"{config.framenet_path}/frame/"):
        if x.endswith(".xml") and x not in set(["Time_vector.xml"]): # skip time_vector, its bad apparently?
            with open(f"{config.framenet_path}/frame/{x}", 'r') as file:
                xml_string = file.read().replace('&gt;', '>').replace('&lt;', '<') 
            
            frame_root = ET.fromstring(xml_string)
            examples = frame_root.findall(".//{http://framenet.icsi.berkeley.edu}ex")
            frame_name = frame_root.attrib["name"]
            
            # each example sentence
            for ex in examples:
                example_toks = [globalvars.tokenizer.cls_token_id]
                example_label = [-100]
                
                # each tag in example sentence
                for piece in ex.iter():
                    if piece.text == None or piece.text == "" or piece.text == " ":
                        continue
                    
                    piece_toks = globalvars.tokenizer.encode(piece.text)[1:-1]
                    example_toks += piece_toks
                    
                    if piece.tag == "{http://framenet.icsi.berkeley.edu}t":
                        example_label += frame_encoder.transform([frame_name]).tolist() * len(piece_toks)
                    else:
                        example_label += [-100] * len(piece_toks)
                    
                    if piece.tail != None and piece.tail.strip() != "":
                        tail_toks = globalvars.tokenizer.encode(piece.tail)[1:-1]
                        example_toks += tail_toks
                        example_label += [-100] * len(tail_toks)
                
                example_toks += [globalvars.tokenizer.sep_token_id]
                example_label += [-100]
                
                if len(example_toks) >= 10:
                    target_examples.append({"toks":example_toks, "labels":example_label})

    return target_examples

def load_example_target_mask(args):
    # Get frame information
    target_examples = []
    
    # Load examples
    for x in os.listdir(f"{config.framenet_path}/frame/"):
        if x.endswith(".xml") and x not in set(["Time_vector.xml"]): # skip time_vector, its bad apparently?
            with open(f"{config.framenet_path}/frame/{x}", 'r') as file:
                xml_string = file.read().replace('&gt;', '>').replace('&lt;', '<') 
            
            frame_root = ET.fromstring(xml_string)
            examples = frame_root.findall(".//{http://framenet.icsi.berkeley.edu}ex")

            # each example sentence
            for ex in examples:
                example_toks = [globalvars.tokenizer.cls_token_id]
                example_label = [-100]
                
                # each tag in example sentence
                for piece in ex.iter():
                    if piece.text == None or piece.text == "" or piece.text == " ":
                        continue
                    
                    piece_toks = globalvars.tokenizer.encode(piece.text)[1:-1]
                    example_toks += piece_toks
                    
                    if piece.tag == "{http://framenet.icsi.berkeley.edu}t":
                        example_label += [1] * len(piece_toks)
                    else:
                        example_label += [-100] * len(piece_toks)
                    
                    if piece.tail != None and piece.tail.strip() != "":
                        tail_toks = globalvars.tokenizer.encode(piece.tail)[1:-1]
                        example_toks += tail_toks
                        example_label += [-100] * len(tail_toks)
                
                example_toks += [globalvars.tokenizer.sep_token_id]
                example_label += [-100]
                
                if len(example_toks) >= 10 and 1 in example_label:
                    target_examples.append({"toks":example_toks, "labels":example_label})

    return target_examples

def load_all_targets():
    from transformers import BertConfig
    # Create config and pass the frame information to model
    model_config = BertConfig.from_pretrained("bert-base-uncased", do_lower_case=True)
    
    # Set global variables
    globalvars.model_config = model_config
    globalvars.max_partition_toks = globalvars.max_token_len
    
    # Create data loader
    data_loader = FrameNetDataLoader(globalvars.tokenizer, None, None)

    # Parse fulltext data
    if os.path.isfile(f"{config.cache_dir}target_dataset.pickle"):
        with open(f"{config.cache_dir}target_dataset.pickle", "rb") as f:
            target_dataset = pickle.load(f)
    else:
        target_dataset = data_loader.get_all_targets()
        with open(f"{config.cache_dir}target_dataset.pickle", "wb") as f:
            pickle.dump(target_dataset, f, pickle.HIGHEST_PROTOCOL)
    
    merged_target_dataset = {}
    
    for x in target_dataset:
        sent = x["sentence"]
        targ = x["target"]
        if sent not in merged_target_dataset:
            merged_target_dataset[sent] = []
        
        merged_target_dataset[sent].append(targ)
    
    target_dataset = []
    
    for sent, targs in merged_target_dataset.items():
        if min([len(x) for x in targs]) != max([len(x) for x in targs]):
            print(sent) 
            continue
        _toks = torch.tensor(globalvars.tokenizer.encode(sent))
        _targ = torch.tensor(np.vstack(targs).max(axis=0))
        
        target_dataset.append({"tokens":_toks, "targets":_targ, "sentence":sent})
        
    return target_dataset


############ incomplete

def load_lu_sentences(args):
    # Get frame information
    frame_info = get_frame_info(f"{config.framenet_path}/frame/")

    # Create frame label encoder to convert frame name -> label
    frame_encoder = LabelEncoder().fit([x["name"] for x in frame_info])

    # Create FE label encoder to convert FE name -> label, for each frame
    fe_encoders = {frame["name"]:LabelEncoder().fit(frame["fes"]) for frame in frame_info}
    
    # when loading, need to keep partitions
    # when training, only train for positive samples, i.e. labels should be -100 or a FE
    
    frame_info = []
    full_fe_examples = {}
    bad_files = set(["lu10604.xml", "lu16043.xml", "lu16423.xml", "lu17223.xml", "lu17711.xml", 
                     "lu17786.xml", "lu17794.xml", "lu17798.xml", "lu17802.xml", "lu17804.xml", 
                     "lu17808.xml", "lu17848.xml", "lu17850.xml", "lu17871.xml", "lu17888.xml", 
                     "lu18266.xml", "lu18267.xml", "lu18275.xml", "lu18292.xml", "lu18667.xml"])
    
    # Load examples
    for x in os.listdir(f"./data/fndata-1.7/lu/"):
        if x.endswith(".xml") and x not in bad_files: # skip time_vector, its bad apparently?
            
            with open(f"./data/fndata-1.7/lu/{x}", 'r') as file:
                xml_string = file.read().replace('&gt;', '>').replace('&lt;', '<') 
            frame_root = ET.fromstring(xml_string)
            frame = frame_root.attrib["frame"]
            print(frame)
            continue
            
            fes = frame_root.findall(".//{http://framenet.icsi.berkeley.edu}FE")
            fe_abbrev = {}
            for fe in fes:
                if "abbrev" in fe.attrib and fe.attrib["abbrev"] not in [None, ""]:
                    fe_abbrev[fe.attrib["abbrev"].lower()] = fe.attrib["name"]
            
            examples = frame_root.findall(".//{http://framenet.icsi.berkeley.edu}ex")
                        
            fe_examples = []
            
            for ex in examples:
                fe_parts = [(ex.text, None)] if ex.text != None and ex.text != ' ' and ex.text != '' else []
                
                for piece in ex:
                    piece_fe = piece.attrib.get("name", None)
                    
                    if piece.text == None or piece.text == "" or piece.text == " ":
                        continue
                    
                    if len(fe_parts) > 0 and piece_fe == fe_parts[-1][1]:
                        fe_parts[-1] = (fe_parts[-1][0] + piece.text, piece_fe)
                    else:
                        if piece_fe != None:
                            fe_parts.append((piece.text, fe_abbrev.get(piece_fe.lower(), piece_fe)))
                        else:
                            fe_parts.append((piece.text, fe_abbrev.get(piece_fe, piece_fe)))
                        
                    if piece == ex[-1] and piece.tail != None:
                        fe_parts.append((piece.tail, None))
                
                fe_examples.append(fe_parts)

            fe_names = [x.attrib["name"] for x in fes]
            frame_info.append({"name":frame_root.attrib["name"], "fes":["[NONFETOKEN]"] + fe_names, "num_elements":len(fe_names)+1})
            full_fe_examples[frame_root.attrib["name"]] = fe_examples
    
    
    fe_example_dataset = []
    # Parse examples into dataset
    for frame, examples in full_fe_examples.items():
        
        for example in examples:
            sent_text = " ".join([x[0].strip() for x in example if x[0] != None])
            sent_toks = globalvars.tokenizer.encode(sent_text)
            
            _fe_label = [-100]
            
            for span_text, span_fe in example:
                span_toks = globalvars.tokenizer.encode(span_text)[1:-1]
                
                if span_fe == None:
                    _fe_label += [-100] * len(span_toks)
                else:
                    if span_fe in fe_encoders[frame].classes_:
                        _fe_label += [fe_encoders[frame].transform([span_fe])[0]] * len(span_toks)
                    else:
                        continue
                    
            _fe_label += [-100]
            
            if len(sent_toks) == len(_fe_label) and len(sent_text) > 30: # basic filter out garbage
                #{"sentence":sent_text, "sentence_tokens":tokenized_sent, "frame":frame_name, 
                # "fes":fe_label if len(fe_spans) > 0 else None}
                fe_example_dataset.append({"sentence":sent_text, "sentence_tokens":sent_toks, 
                                           "frame":frame, "fes":_fe_label})
    
    # Make tensors from dataset
    train_example_fes = pad_sequences([x["fes"] for x in fe_example_dataset], maxlen=globalvars.max_token_len, value=-100, padding="post")
    train_example_toks = pad_sequences([x["sentence_tokens"] for x in fe_example_dataset], maxlen=globalvars.max_token_len, padding="post")
    train_example_frames = np.array([x["frame"] for x in fe_example_dataset])
    
    return (torch.tensor(train_example_toks), train_example_frames, torch.tensor(train_example_fes))

def load_dataset_with_pos_encoding(args):
    # Get frame information
    frame_info = get_frame_info(f"{config.framenet_path}/frame/")

    # Create frame label encoder to convert frame name -> label
    frame_encoder = LabelEncoder().fit([x["name"] for x in frame_info])

    # Create FE label encoder to convert FE name -> label, for each frame
    fe_encoders = {frame["name"]:LabelEncoder().fit(frame["fes"]) for frame in frame_info}

    # Create config and pass the frame information to model
    model_config = FrameBertConfig.from_pretrained("bert-base-cased", supported_frames=frame_info)
    
    # Set global variables
    globalvars.frame_encoder = frame_encoder
    globalvars.fe_encoders = fe_encoders
    globalvars.model_config = model_config
    
    # Define dataset files
    if args.semafor:
        train_dataset_files = [f"{config.framenet_path}/fulltext/{x}" for x in config.OPENSESAME_TRAIN_FILES]
        test_dataset_files = [f"{config.framenet_path}/fulltext/{x}" for x in config.OPENSESAME_TEST_FILES]
    else:
        print("This is currently unsupported.")
        return None
    
    # Create data loader
    data_loader = FrameNetDataLoader(globalvars.tokenizer, fe_encoders, frame_encoder)

    # Parse fulltext data
    if args.evaluate and args.train:
        if os.path.isfile(f"{config.cache_dir}train_dataset_pos_enc.pickle"):
            with open(f"{config.cache_dir}train_dataset_pos_enc.pickle", "rb") as f:
                train_dataset = pickle.load(f)
        else:
            train_dataset = data_loader.get_fulltext_data_with_pos(train_dataset_files)
            with open(f"{config.cache_dir}train_dataset_pos_enc.pickle", "wb") as f:
                pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
            
        if os.path.isfile(f"{config.cache_dir}test_dataset_pos_enc.pickle"):
            with open(f"{config.cache_dir}test_dataset_pos_enc.pickle", "rb") as f:
                test_dataset = pickle.load(f)
        else:
            test_dataset = data_loader.get_fulltext_data_with_pos(test_dataset_files)
            with open(f"{config.cache_dir}test_dataset_pos_enc.pickle", "wb") as f:
                pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)

    elif args.evaluate:
        if os.path.isfile(f"{config.cache_dir}test_dataset_pos_enc.pickle"):
            with open(f"{config.cache_dir}test_dataset_pos_enc.pickle", "rb") as f:
                test_dataset = pickle.load(f)
        else:
            test_dataset = data_loader.get_fulltext_data_with_pos(test_dataset_files)
            with open(f"{config.cache_dir}test_dataset_pos_enc.pickle", "wb") as f:
                pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)
            
    else:
        assert False, "--train without --evaluate is currently unsupported"
    
    if args.train:
        train_toks, train_fes, train_targets, train_frames, train_sents = data_loader.pad_datasets(train_dataset, max_len=globalvars.max_token_len)
        
        train_toks = torch.tensor(train_toks)
        train_fes = torch.tensor(train_fes)
        train_targets = torch.tensor(train_targets)
        train_frames = np.array(train_frames)
        train_sents = np.array(train_sents)

    test_toks, test_fes, test_targets, test_frames, test_sents = data_loader.pad_datasets(test_dataset, max_len=globalvars.max_token_len)
    
    test_toks = torch.tensor(test_toks)
    test_fes = torch.tensor(test_fes)
    test_targets = torch.tensor(test_targets)
    test_frames = np.array(test_frames)
    test_sents = np.array(test_sents)
    
    if args.train:
        return ((train_toks, train_fes, train_targets, train_frames, train_sents), (test_toks, test_fes, test_targets, test_frames, test_sents), model_config)
    elif args.evaluate:
        return (test_toks, test_fes, test_targets, test_frames, test_sents)
