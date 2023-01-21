from transformers import BertTokenizer
import torch

import config

# Global variables
tokenizer = BertTokenizer.from_pretrained(config.bert_version, do_lower_case=True if config.bert_version.endswith("uncased") else False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

frame_encoder = None
fe_encoders = None
model_config = None
max_token_len = 168
max_partition_toks = 16