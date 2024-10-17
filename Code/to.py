import torch
from datasets import load_dataset, Dataset
import glob
from transformers import GPT2Tokenizer



def tokenize():
    seed = 42
    torch.manual_seed(seed)
    medical_files = glob.glob('datasets/Token-Medical-Shot/train-*.parquet')
    # selected_files = random.sample(medical_files, int(0.5 * len(medical_files)))
    dataset1 = load_dataset('parquet', data_files=medical_files, num_proc=32)
    dataset2 = load_dataset("Skylion007/openwebtext", split="train[0:15%]",cache_dir="./cache",num_proc = 32,trust_remote_code=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    tokenized_datasets1 = dataset1['train'].map(tokenize_function, batched=True,num_proc = 64)
    tokenized_datasets1.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokenized_datasets1.save_to_disk('datasets/tokenized')

if __name__ == '__main__':
    tokenize()

