import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from utilities import tokenize
from plot_utilities import plot_confusion_matrix, plot_roc_curve
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model,criterion,optimizer,scheduler,epoch,dataloader):
    model.train()
    running_loss=0.
    acc=0
    total_samples=0
    for batch_data, batch_ids,batch_seq_ids, batch_labels in tqdm(dataloader, desc="Processing Batches"):
        batch_data,batch_ids,batch_seq_ids, batch_labels  = batch_data.to(device), batch_ids.to(device),batch_seq_ids.to(device), batch_labels.to(device)
        Y_prob, Y_hat, A, A_2,_ =model(batch_data,batch_ids,batch_seq_ids)
        Y_prob=Y_prob.squeeze(1)
        Y_hat = Y_hat.view_as(batch_labels)
        loss = criterion(Y_prob, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        acc += ((Y_hat == batch_labels).sum().item())
        total_samples += batch_labels.size(0)  
    running_loss /= len(dataloader)
    print(f'Epoch: {epoch}, Loss: {running_loss:.4f}, LR: {scheduler.get_last_lr()}')
    acc=acc/total_samples*100
    print(f'train accurecy: {acc:.1f}%')
    return running_loss

def test(model,dataloader):
    model.eval()
    acc=0
    total_samples=0
    output=[]
    true_lables=[]
    with torch.no_grad():
        for batch_data, batch_ids,batch_seq_ids, batch_labels in tqdm(dataloader, desc="Processing Batches"):
            batch_data,batch_ids,batch_seq_ids, batch_labels  = batch_data.to(device), batch_ids.to(device),batch_seq_ids.to(device), batch_labels.to(device)
            Y_prob, Y_hat, A, A_2,_ =model(batch_data,batch_ids,batch_seq_ids)         
            output += Y_hat.cpu().tolist()
            true_lables += batch_labels.cpu().tolist()
            Y_prob=Y_prob.squeeze(1)
            Y_hat = Y_hat.view_as(batch_labels)
            acc += ((Y_hat == batch_labels).sum().item())
            total_samples += batch_labels.size(0)  # Track the total number of samples processed

    acc=acc/total_samples*100
    print(f'acc: {acc:.1f}%')
    print("\nClassification Report:")
    print(classification_report(true_lables, output, target_names=["Human", "Animal"]))

    plot_confusion_matrix(true_lables, output)
    plot_roc_curve(true_lables, output)
    return output,true_lables

def test_one_virus(model,ft_model,data,ids,seq_ids,labels=None):
    model.eval()
    if not labels:
        labels=np.zeros(len(ids))
    embeddings, ids,seq_ids, labels=tokenize(data,ids,seq_ids,labels,ft_model)
    with torch.no_grad():
        Y_prob, Y_hat, A, A_2,_ =model(embeddings,ids,seq_ids)         
    return Y_prob, Y_hat, A, A_2 