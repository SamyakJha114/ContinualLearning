import argparse
import torch
import os
import torch.nn as nn
from models.CAT import CAT
from models.GPTdemix import GPTDeMIX
from utils.utils import evaluate_perplexity, train, save_model
from DataCode.preprocessing_data import load_and_preprocess_datasets

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GPT-DeMIX model.")
    parser.add_argument('--model', type=str, default='CAT', help='Kind of model for the experiment')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--embed_size', type=int, default=256, help='Embedding size for the model')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads in the model')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers in the model')
    parser.add_argument('--num_domains', type=int, default=3, help='Number of domains in the model')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--save_path', type=str, default='Results', help='Path to save the model')
    parser.add_argument('--scheduler', type=str, default='none', help='Scheduler to schedule the training')
    parser.add_argument('--gpu', type=int, help='The index of the GPU to be used')

    args = parser.parse_args()

    # Seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Load datasets
    train_dataloader_domain_1, test_dataset1, vocab_size, train_dataloader_domain_2, test_dataset2, train_dataloader_domain_3, test_dataset3 = load_and_preprocess_datasets(args.batch_size)

    # Initialize the model
    if args.model == 'CAT':
        model = CAT(args.embed_size, args.heads, args.num_layers, args.num_domains, vocab_size, args.max_length)
    else:
        model = GPTDeMIX(args.embed_size, args.heads, args.num_layers, args.num_domains, vocab_size, args.max_length)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Dataloaders for different domains
    dataloaders = [train_dataloader_domain_1, train_dataloader_domain_2, train_dataloader_domain_3]

    # Device configuration
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(args.epochs):
        train(model, dataloaders, optimizer, criterion, num_epochs=1, device=device, vocab_size=vocab_size, scheduler_type=args.scheduler)

        perplexity1 = evaluate_perplexity(args.model, model, test_dataset1, criterion, device, domain_idx=0)
        perplexity2 = evaluate_perplexity(args.model, model, test_dataset2, criterion, device, domain_idx=1)
        perplexity3 = evaluate_perplexity(args.model, model, test_dataset3, criterion, device, domain_idx=2)

        output1 = f'Epoch [{epoch+1}/{args.epochs}], Test Perplexity (Domain 1): {perplexity1:.4f}'
        output2 = f'Epoch [{epoch+1}/{args.epochs}], Test Perplexity (Domain 2): {perplexity2:.4f}'
        output3 = f'Epoch [{epoch+1}/{args.epochs}], Test Perplexity (Domain 3): {perplexity3:.4f}'

        # Save the model
        save_model(args.model, model, optimizer, 'datasets')

        output_file_path = os.path.join(args.save_path, 'output.txt')

        print(output1)
        print(output2)
        print(output3)

        with open(output_file_path, 'a') as f:
            f.write(args.model + ' result :- \n')
            f.write(output1 + '\n')
            f.write(output2 + '\n')
            f.write(output3 + '\n')

if __name__ == '__main__':
    main()
