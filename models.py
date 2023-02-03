from transformers import BertPreTrainedModel, BertModel
from transformers import DebertaPreTrainedModel, DebertaModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.nn import CrossEntropyLoss

import torch

import globalvars
import config
from torch.utils.data import Dataset

from typing import List, Dict

# Different Aggregation methods
class FrameElementIdentifier(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # supported_frames = n-dim array of {frame_name/id:"", num_fes:#} dicts
    def __init__(self, config, fe_weights=None):

        super().__init__(config)

        assert self.config.supported_frames != None, "Please include the supported frames"

        self.frames = self.config.supported_frames

        self.bert = BertModel(config, add_pooling_layer=True).to(globalvars.device)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.relu = torch.nn.ReLU()

        self.fe_classifiers = {frame["name"]:torch.nn.Linear(config.hidden_size, frame["num_elements"]).to(globalvars.device) for frame in self.frames}

        self.loss_fct = CrossEntropyLoss()
        self.fe_weights = fe_weights

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, 
                partitions=None, frames=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state

        # Get partitioned outputs
        sequence_output = [torch.stack([sequence_output[i, partitions[i] == p, :].mean(axis=0) for p in range(partitions[i].max()+1)]) for i in range(sequence_output.shape[0])]

        sequence_output = [self.fe_classifiers[frame](x) for x, frame in zip(sequence_output, frames)]

        if labels != None:
            if len([x for x in labels if x.shape[0] > 1]) > 0:
                losses = sum([self.loss_fct(sequence_output[i], labels[i]) for i in range(len(labels)) if labels[i].shape[0] > 1]) / len([x for x in labels if x.shape[0] > 1])
            else:
                losses = None
        else:
            losses = None

        # concatenate vs add frame to logits
        # 2 output layers probably important
        # Pass frame into bert as token_type 1 with sentence

        return TokenClassifierOutput(
            loss=losses,
            logits=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class FrameElementIdentifierAggFirst(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # supported_frames = n-dim array of {frame_name/id:"", num_fes:#} dicts
    def __init__(self, config, fe_weights=None):

        super().__init__(config)

        assert self.config.supported_frames != None, "Please include the supported frames"

        self.frames = self.config.supported_frames

        self.bert = BertModel(config, add_pooling_layer=True).to(globalvars.device)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.relu = torch.nn.ReLU()

        self.fe_classifiers = {frame["name"]:torch.nn.Linear(config.hidden_size, frame["num_elements"]).to(globalvars.device) for frame in self.frames}

        self.loss_fct = CrossEntropyLoss()
        self.fe_weights = fe_weights

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, 
                partitions=None, frames=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state

        # Get partitioned outputs
        sequence_output = [torch.stack([sequence_output[i, partitions[i] == p, :][0]
                                        for p in range(partitions[i].max()+1)]) 
                           for i in range(sequence_output.shape[0])]
        
        sequence_output = [self.fe_classifiers[frame](x) for x, frame in zip(sequence_output, frames)]

        if labels != None:
            losses = sum([self.loss_fct(sequence_output[i], labels[i]) for i in range(len(labels)) if labels[i].shape[0] > 1]) / len([x for x in labels if x.shape[0] > 1])
        else:
            losses = None

        # concatenate vs add frame to logits
        # 2 output layers probably important
        # Pass frame into bert as token_type 1 with sentence

        return TokenClassifierOutput(
            loss=losses,
            logits=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class FrameElementIdentifierAggSum(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # supported_frames = n-dim array of {frame_name/id:"", num_fes:#} dicts
    def __init__(self, config, fe_weights=None):

        super().__init__(config)

        assert self.config.supported_frames != None, "Please include the supported frames"

        self.frames = self.config.supported_frames

        self.bert = BertModel(config, add_pooling_layer=True).to(globalvars.device)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.relu = torch.nn.ReLU()

        self.fe_classifiers = {frame["name"]:torch.nn.Linear(config.hidden_size, frame["num_elements"]).to(globalvars.device) for frame in self.frames}

        self.loss_fct = CrossEntropyLoss()
        self.fe_weights = fe_weights

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, 
                partitions=None, frames=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state

        # Get partitioned outputs
        sequence_output = [torch.stack([sequence_output[i, partitions[i] == p, :].sum(axis=0)
                                        for p in range(partitions[i].max()+1)]) 
                           for i in range(sequence_output.shape[0])]
        
        sequence_output = [self.fe_classifiers[frame](x) for x, frame in zip(sequence_output, frames)]

        if labels != None:
            losses = sum([self.loss_fct(sequence_output[i], labels[i]) for i in range(len(labels)) if labels[i].shape[0] > 1]) / len([x for x in labels if x.shape[0] > 1])
        else:
            losses = None

        # concatenate vs add frame to logits
        # 2 output layers probably important
        # Pass frame into bert as token_type 1 with sentence

        return TokenClassifierOutput(
            loss=losses,
            logits=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class FrameElementIdentifierAggMax(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # supported_frames = n-dim array of {frame_name/id:"", num_fes:#} dicts
    def __init__(self, config, fe_weights=None):

        super().__init__(config)

        assert self.config.supported_frames != None, "Please include the supported frames"

        self.frames = self.config.supported_frames

        self.bert = BertModel(config, add_pooling_layer=True).to(globalvars.device)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.relu = torch.nn.ReLU()

        self.fe_classifiers = {frame["name"]:torch.nn.Linear(config.hidden_size, frame["num_elements"]).to(globalvars.device) for frame in self.frames}

        self.loss_fct = CrossEntropyLoss()
        self.fe_weights = fe_weights

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, 
                partitions=None, frames=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state

        # Get partitioned outputs
        sequence_output = [torch.stack([sequence_output[i, partitions[i] == p, :].max(dim=0).values
                                        for p in range(partitions[i].max()+1)]) 
                           for i in range(sequence_output.shape[0])]
        
        sequence_output = [self.fe_classifiers[frame](x) for x, frame in zip(sequence_output, frames)]

        if labels != None:
            losses = sum([self.loss_fct(sequence_output[i], labels[i]) for i in range(len(labels)) if labels[i].shape[0] > 1]) / len([x for x in labels if x.shape[0] > 1])
        else:
            losses = None

        # concatenate vs add frame to logits
        # 2 output layers probably important
        # Pass frame into bert as token_type 1 with sentence

        return TokenClassifierOutput(
            loss=losses,
            logits=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# W/ FE definitions
class FrameElementIdentifierUsingFEDefs(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # supported_frames = n-dim array of {frame_name/id:"", num_fes:#} dicts
    def __init__(self, config, fe_weights=None):

        super().__init__(config)

        assert self.config.supported_frames != None, "Please include the supported frames"

        self.frames = self.config.supported_frames

        self.bert = BertModel(config, add_pooling_layer=True).to(globalvars.device)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.relu = torch.nn.ReLU()

        self.fe_classifier = torch.nn.Linear(config.hidden_size, 2)

        self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]))

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state

        # Get partitioned outputs
        # [CLS] + input_sent + [SEP] + fe_def + [SEP]
        # [CLS] + input_sent + [SEP] + frame_name + fe_def + [SEP]
        # [CLS] + input_sent + [SEP] + frame_def + fe_def + [SEP]
        # consider full fe_def embedding vs embedding of fe word in fe_def
        sequence_output = self.fe_classifier(sequence_output)
        
        if labels != None:
            losses = sum([self.loss_fct(sequence_output[i], labels[i]) for i in range(labels.shape[0])]) / labels.shape[0]
        else:
            losses = None

        # concatenate vs add frame to logits
        # 2 output layers probably important
        # Pass frame into bert as token_type 1 with sentence

        return TokenClassifierOutput(
            loss=losses,
            logits=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# DeBERTa model
class DebertaFrameElementIdentifier(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # supported_frames = n-dim array of {frame_name/id:"", num_fes:#} dicts
    def __init__(self, config, fe_weights=None):

        super().__init__(config)

        assert self.config.supported_frames != None, "Please include the supported frames"

        self.frames = self.config.supported_frames

        self.bert = DebertaModel(config).to(globalvars.device)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.fe_classifiers = {frame["name"]:torch.nn.Linear(config.hidden_size, frame["num_elements"]).to(globalvars.device) for frame in self.frames}

        self.loss_fct = CrossEntropyLoss()

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, 
                partitions=None, frames=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state
        
        # Get partitioned outputs
        sequence_output = [torch.stack([sequence_output[i, partitions[i] == p, :].mean(axis=0) for p in range(partitions[i].max()+1)]) for i in range(sequence_output.shape[0])]

        sequence_output = [self.fe_classifiers[frame](x) for x, frame in zip(sequence_output, frames)]
        
        if labels != None:
            if len([x for x in labels if x.shape[0] > 1]) > 0:
                losses = sum([self.loss_fct(sequence_output[i], labels[i]) for i in range(len(labels)) if labels[i].shape[0] > 1]) / len([x for x in labels if x.shape[0] > 1])
            else:
                losses = None
        else:
            losses = None

        # concatenate vs add frame to logits
        # 2 output layers probably important
        # Pass frame into bert as token_type 1 with sentence

        return TokenClassifierOutput(
            loss=losses,
            logits=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# POS-aware model (incomplete)
class FrameElementIdentifierWithPOS(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # supported_frames = n-dim array of {frame_name/id:"", num_fes:#} dicts
    def __init__(self, config, fe_weights=None):

        super().__init__(config)

        assert self.config.supported_frames != None, "Please include the supported frames"

        self.frames = self.config.supported_frames

        self.bert = BertModel(config, add_pooling_layer=True).to(globalvars.device)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.relu = torch.nn.ReLU()

        self.fe_classifiers = {frame["name"]:torch.nn.Linear(config.hidden_size, frame["num_elements"]).to(globalvars.device) for frame in self.frames}

        self.loss_fct = CrossEntropyLoss()
        self.fe_weights = fe_weights

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, 
                partitions=None, frames=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state
        

        # Get partitioned outputs
        sequence_output = [torch.stack([sequence_output[i, partitions[i] == p, :].mean(axis=0) for p in range(partitions[i].max()+1)]) for i in range(sequence_output.shape[0])]

        sequence_output = [self.fe_classifiers[frame](x) for x, frame in zip(sequence_output, frames)]

        if labels != None:
            if len([x for x in labels if x.shape[0] > 1]) > 0:
                losses = sum([self.loss_fct(sequence_output[i], labels[i]) for i in range(len(labels)) if labels[i].shape[0] > 1]) / len([x for x in labels if x.shape[0] > 1])
            else:
                losses = None
        else:
            losses = None

        # concatenate vs add frame to logits
        # 2 output layers probably important
        # Pass frame into bert as token_type 1 with sentence

        return TokenClassifierOutput(
            loss=losses,
            logits=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Target Identifier
class CandidateTargetClassifier(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):

        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.relu = torch.nn.ReLU()

        self.classifier = torch.nn.Linear(config.hidden_size, 2) # 0 or 1
        
        self.loss_fct = CrossEntropyLoss()
        
        self._device = globalvars.device

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """Predict whether target spans are valid

        Args:
            input_ids (Tensor(b, n), optional): _description_.
            labels (Tensor(b, t, n), optional): Batch containing b samples of t n-length candidate targets
        """

        assert input_ids.shape[0] == 1, "unsupported for batch size > 1"

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state # (b, n, H)
        logits = []

        target_span = labels != -100 # (b, t, n) ~ one-hot spans of candidate target tokens

        # for each sample in batch 
        for i in range(input_ids.shape[0]):
            # For each span, represented by subsequent 0/1s in a row in label,
            # merge tokens and predict
            for n in range(labels[i].shape[0]):
                cur_span = target_span[i][n]
                logits.append(self.classifier(sequence_output[i,cur_span,:].mean(dim=0))) # (2,)

        logits = torch.vstack(logits).transpose(0,1).unsqueeze(0).to(self.device) # (b*t, 2)
        loss = None

        if labels != None and len(sequence_output) > 0 and self.training:
            # target_labels = torch.tensor([x.unique()[-1] for x in labels[0]]).unsqueeze(0).to(self.device)
            target_labels = labels.max(dim=-1).values.to(self.device)
            loss = self.loss_fct(logits, target_labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# FE classifier using candidate spans
class CandidateFEClassifier(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, fe_weights=None):

        super().__init__(config)

        assert self.config.supported_frames != None, "Please include the supported frames"

        self.frames = self.config.supported_frames

        self.bert = BertModel(config, add_pooling_layer=True).to(globalvars.device)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.relu = torch.nn.ReLU()

        self.fe_classifiers = {frame["name"]:torch.nn.Linear(config.hidden_size, frame["num_elements"]).to(globalvars.device) for frame in self.frames}

        self.loss_fct = CrossEntropyLoss()
        self.fe_weights = fe_weights

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, 
                argument_spans=None, frame=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert input_ids.shape[0] == 1, "currently only supporting batch size = 1"

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state
        
        base_logits = self.fe_classifiers[frame](sequence_output) # classify each token, w/o any grouping

        # Get partitioned outputs
        sequence_output = sequence_output.repeat((argument_spans.shape[0], 1, 1))
        sequence_output[argument_spans != 1] = 0
        sequence_output = sequence_output.sum(dim=1) / argument_spans.sum(dim=1).unsqueeze()
        sequence_output = self.fe_classifiers[frame](sequence_output)
        
        sequence_output = sequence_output + base_logits

        if labels != None:
            losses = self.loss_fct(sequence_output, labels.squeeze())
        else:
            losses = None

        del base_logits
        return TokenClassifierOutput(
            loss=losses,
            logits=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )