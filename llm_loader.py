# -*- coding:utf-8 -*-

from inputters.dataset import LM4MTDataset
from models.llmhub import LlamaCLForMSP, BloomCLForMSP
from transformers import LlamaTokenizer, AutoTokenizer

MODE = {"llama": {"model": LlamaCLForMSP, "tokenizer": LlamaTokenizer,
                "dataset": LM4MTDataset, 
                "default_special _tokens": {"DEFAULT_PAD_TOKEN" : "[PAD]", "DEFAULT_EOS_TOKEN": "</s>", "DEFAULT_BOS_TOKEN": "<s>", "DEFAULT_UNK_TOKEN": "<unk>"}}, 
        "bloom": {"model": BloomCLForMSP, "tokenizer": AutoTokenizer,
                "dataset": LM4MTDataset,
                "default_special _tokens": {"DEFAULT_PAD_TOKEN" : "<pad>", "DEFAULT_EOS_TOKEN": "</s>", "DEFAULT_BOS_TOKEN": "<s>", "DEFAULT_UNK_TOKEN": "<unk>"}}, 
        }