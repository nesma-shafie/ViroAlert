import torch
import torch.nn as nn
from constant import char2idx, idx2char, max_smile_len, device,vocab_size
from torch.utils.data import DataLoader
import tqdm 
from torch.cuda.amp import autocast
import gc
import os
import numpy as np
from SMILESDataset import SMILESDataset, collate_fn
def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,g_scaler,d_scaler, epoch, checkpoint_dir='/kaggle/working'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')    
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_scalar_state_dict': g_scaler.state_dict(),
        'd_scalar_state_dict': d_scaler.state_dict(),
        'epoch': epoch
    }, checkpoint_path)
    
    print(f"Saved checkpoint: {checkpoint_path}")
def load_checkpoint(generator, discriminator, g_optimizer, d_optimizer,g_scaler,d_scaler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)  # ensure correct device
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    g_scaler.load_state_dict(checkpoint['g_scalar_state_dict'])
    d_scaler.load_state_dict(checkpoint['d_scalar_state_dict'])

    epoch = checkpoint['epoch']  
    print(f"Checkpoint loaded from '{checkpoint_path}' at epoch {epoch}")
    return epoch

def train_discriminator(discriminator, dataloader, generator, d_optimizer, d_scaler, max_len=128,d_criterion=None):
    total_loss = 0
    for real_batch in dataloader:
        real_batch = real_batch.to(device)
        d_optimizer.zero_grad()
        with autocast():
            real_output = discriminator(real_batch)
            with torch.no_grad():
                fake_seqs, _ = generator.sample(char2idx['<SOS>'],max_len=max_len,batch_size=real_batch.size(0))
            fake_seq_tensor = fake_seqs.to(device)
            fake_output = discriminator(fake_seq_tensor)
            real_labels = torch.ones_like(real_output)
            fake_labels = torch.zeros_like(fake_output)
            d_loss_real = d_criterion(real_output, real_labels)
            d_loss_fake = d_criterion(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake
        d_scaler.scale(d_loss).backward()
        d_scaler.step(d_optimizer)
        d_scaler.update()
        total_loss += d_loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"train Discriminator Loss: {avg_loss:.4f}")
    torch.cuda.empty_cache()
    gc.collect()

def train_generator(generator, dataloader,g_criterion,  g_optimizer=None,g_scaler=None):
     total_loss=0
     for batch in dataloader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:].contiguous().view(-1)
            with autocast():
                log_probs, _ = generator(inputs)
                loss = g_criterion(log_probs.view(-1, vocab_size), targets)
                total_loss+=loss.item()
            g_scaler.scale(loss).backward()
            g_scaler.step(g_optimizer)
            g_scaler.update()
            g_optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear memory after each epoch
            gc.collect()
     print(f"Pre-train generator Loss: {total_loss/len(dataloader):.4f}")
    
def pre_train(generator, discriminator,g_optimizer,d_optimizer,g_scaler, d_scaler,real_data, max_len=128):
    k_epochs=10
    d_steps=10
    batch_size = 1024
    dataset = SMILESDataset(real_data, char2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    g_criterion = nn.CrossEntropyLoss(ignore_index=char2idx["<PAD>"])
    # Pre-train generator with MLE
    for _ in tqdm(range(k_epochs)):
       train_generator(generator, dataloader, g_criterion, g_optimizer, g_scaler)  # Pre-train
       print("end  generator ")
    # Pre-train discriminator 
    d_criterion = torch.nn.BCEWithLogitsLoss()
    for _ in tqdm(range(d_steps)):
        train_discriminator(discriminator, dataloader, generator, d_optimizer, d_scaler, max_len, d_criterion)
    print("end discriminator")
def main_train(generator, discriminator,g_optimizer,d_optimizer,g_scaler, d_scaler,real_data,start_point=-1, epochs=50, max_len=128):
    d_steps=10
    g_steps=5
    batch_size = 1024
    dataset = SMILESDataset(real_data, char2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Main training loop
    for epoch in tqdm(range(start_point+1,epochs)):
        # G-steps
        for i in range(g_steps):
            loss_g = generator.train_step(char2idx['<SOS>'], max_len, g_optimizer, discriminator,g_scaler)
            print(f"Epoch {epoch} step{i}, Generator Loss: {loss_g:.4f}")
        
        # D-steps
        for i in range(d_steps):
            train_discriminator(discriminator, dataloader, generator, d_optimizer, d_scaler, max_len)               
        save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,g_scaler,d_scaler,epoch)