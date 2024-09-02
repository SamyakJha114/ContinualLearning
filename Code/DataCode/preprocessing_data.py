import torch
from datasets import load_dataset, Dataset
import os
import random
import glob
from torch.utils.data import DataLoader,random_split
from transformers import GPT2Tokenizer
from utils.utils import read_text_files

def load_and_preprocess_datasets(batch_size):
    seed = 42
    torch.manual_seed(seed)
    medical_files = glob.glob('datasets/Token-Medical-Shot/train-*.parquet')
    selected_files = random.sample(medical_files, int(0.5 * len(medical_files)))
    dataset1 = load_dataset('parquet', data_files=selected_files, num_proc=32)
    dataset2 = load_dataset("Skylion007/openwebtext", split="train[0:15%]",cache_dir="./cache",num_proc = 32,trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    
    tokenized_datasets1 = dataset1['train'].map(tokenize_function, batched=True,num_proc = 64)
    tokenized_datasets1.set_format(type="torch", columns=["input_ids", "attention_mask"])

    tokenized_datasets2 = dataset2.map(tokenize_function, batched=True,num_proc = 64)
    tokenized_datasets2.set_format(type="torch", columns=["input_ids", "attention_mask"])

    print(len(tokenized_datasets1))
    train_size = int(0.5 * len(tokenized_datasets1))
    test_size = len(tokenized_datasets1) - train_size
    train_dataset, test_dataset = random_split(tokenized_datasets1, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    print((train_size),(test_size))

    batch_size = batch_size
    train_dataloader_domain_2 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader_domain_2 = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    train_size = int(0.5 * len(tokenized_datasets2))
    test_size = len(tokenized_datasets2) - train_size
    train_dataset, test_dataset = random_split(tokenized_datasets2, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

    train_dataloader_domain_3 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader_domain_3 = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    directory = "datasets/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled"
    file_pattern = "news.en-"
    data = read_text_files(directory, file_pattern)
    data = random.sample(data, int(len(data) * 0.05))
    dataset3 = Dataset.from_dict({"text": data})
    tokenized_datasets3 = dataset3.map(tokenize_function, batched=True,num_proc = 64)
    tokenized_datasets3.set_format(type="torch", columns=["input_ids", "attention_mask"])
    train_dataloader_domain_1 = DataLoader(tokenized_datasets3, batch_size=batch_size, shuffle=True)
    
    directory1 = "datasets/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled"
    file_pattern = "news.en-"
    data = read_text_files(directory, file_pattern)
    data = random.sample(data, int(len(data) * 0.05))
    dataset3 = Dataset.from_dict({"text": data})
    tokenized_datasets3 = dataset3.map(tokenize_function, batched=True,num_proc = 64)
    tokenized_datasets3.set_format(type="torch", columns=["input_ids", "attention_mask"])

    test_dataloader_domain_1 = DataLoader(tokenized_datasets3, batch_size=batch_size, shuffle=True)

    return train_dataloader_domain_1,test_dataloader_domain_1,tokenizer.vocab_size + 1,train_dataloader_domain_2,test_dataloader_domain_2,train_dataloader_domain_3,test_dataloader_domain_3 