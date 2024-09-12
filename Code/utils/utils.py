import torch
import torch.optim.lr_scheduler as lr_scheduler
import os

def evaluate_perplexity(model_name,model, dataloader, criterion, device,domain_idx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = inputs.clone()
            if model_name == 'CAT':
                outputs = model(inputs[:, :-1], domain_idx,training = False)
            else:
                outputs = model(inputs[:, :-1], domain_idx)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels[:, 1:].reshape(-1))
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
    print(f"average_test_loss :- {total_loss / total_tokens}")
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

def save_model(model_name,model, optimizer, path):
    torch.save(model.state_dict(), path + f'/model{model_name}_state.pth')
    torch.save(optimizer.state_dict(),path + f'/optimizer{model_name}_state.pth')

def read_text_files(directory, file_pattern):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(file_pattern)])
    data = []
    for i,file_path in enumerate(files):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(line.strip())
    return data

def train(model, dataloaders, optimizer, criterion, num_epochs, device, model_name , scheduler_type=None, print_every=1000):
    model.train()
    
    # Define the scheduler based on the provided type
    if scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    elif scheduler_type == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
    else:
        scheduler = None

    for domain_id, dataloader in enumerate(dataloaders):
        if model_name == 'CAT':
            if domain_id != 0 :
                model.init_param(domain_id)
        print(f"Training domain :- {domain_id}")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, batch in enumerate(dataloader, 1):
                inputs = batch['input_ids'].to(device)  
                labels = inputs.clone()
                optimizer.zero_grad()
                outputs = model(inputs[:, :-1], domain_id)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels[:, 1:].reshape(-1))
                loss.backward()
                optimizer.step()
                
                # Step the scheduler if it's defined
                if scheduler:
                    scheduler.step()

                running_loss += loss.item()
                
                if i % print_every == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Domain [{domain_id+1}/{len(dataloaders)}], Iteration [{i}], Loss: {loss.item():.4f}')
                    
            avg_loss = running_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Domain [{domain_id+1}/{len(dataloaders)}], Average Loss: {avg_loss:.4f}')
        
