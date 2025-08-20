import torch
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
from utilities import tokenize 
from sklearn.manifold import TSNE
def plot_tsne_virus_embeddings(model, ft_model, sequences, labels, virus_ids, seq_ids, sample_size=1000, save_path=None):
    # Step 1: Sample a random subset of sequences
    n_samples = min(sample_size, len(sequences))
    sample_indices = np.random.choice(len(sequences), n_samples, replace=False)
    sample_sequences = sequences.iloc[sample_indices]
    sample_labels = labels[sample_indices]
    sample_virus_ids = virus_ids[sample_indices]
    sample_seq_ids = seq_ids[sample_indices]
    
    # Step 2: Identify unique viruses in the subset
    unique_virus_ids = np.unique(sample_virus_ids)
    unique_indices = [np.where(sample_virus_ids == vid)[0][0] for vid in unique_virus_ids]
    
    # Step 3: Prepare model for inference
    model.eval()
    embeddings = []
    final_labels = []
    
    # Step 4: Process each unique virus
    for i, virus_idx in enumerate(tqdm(unique_indices, desc="Processing Viruses")):
        # Get data for one representative sequence of the virus
        virus_sequence = sample_sequences.iloc[[virus_idx]]
        virus_id = sample_virus_ids[[virus_idx]]
        virus_seq_id = sample_seq_ids[[virus_idx]]
        virus_label = sample_labels[[virus_idx]]
        # Tokenize the sequence using FastText
        tokenized_data, v_id, s_id, _ = tokenize(virus_sequence, virus_id, virus_seq_id, virus_label, ft_model)
    
        # Run model to get virus-level embedding (output2)
        with torch.no_grad():
            Y_prob, Y_hat, A, A_2, output2 = model(tokenized_data, v_id, s_id)
        
    # Step 5: Convert embeddings and labels to arrays
        embeddings.append(np.squeeze(output2.cpu().numpy()))
        final_labels.append(virus_label[0])
    embeddings = np.array(embeddings)  # Shape: [n_unique_viruses, EMBEDDING_SIZE]
    final_labels = np.array(final_labels)

    # Step 6: Apply t-SNE to reduce embeddings to 2D
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    embeddings_3d = tsne.fit_transform(embeddings)

    fig1 = plt.figure(figsize=(12, 10))
    human_points = final_labels == 0
    animal_points = final_labels == 1
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(embeddings_3d[human_points, 2], embeddings_3d[human_points, 0], embeddings_3d[human_points, 1],
                    c='blue', label='Human', alpha=0.4, s=60)
    ax1.scatter(embeddings_3d[animal_points, 2], embeddings_3d[animal_points, 0], embeddings_3d[animal_points, 1],
                    c='red', label='Animal', alpha=0.4, s=60)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)
    ax1.set_zlim(-25, 25)
    ax1.set_xlabel('t-SNE Component 1', fontsize=12, labelpad=10)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12, labelpad=10)
    ax1.set_zlabel('t-SNE Component 3', fontsize=12, labelpad=10)
    ax1.set_title('t-SNE of Virus-Level Embeddings (Post-Bag-Level Attention) - 3D', fontsize=14, pad=15)
    ax1.legend(fontsize=10, markerscale=2, title='Categories', title_fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    plt.show()
    

def plot_confusion_matrix(true_labels, pred_labels, class_names=["Human", "Animal"]):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def plot_roc_curve(true_labels, pred_labels):
    true_bin = label_binarize(true_labels, classes=[0, 1])
    pred_bin = label_binarize(pred_labels, classes=[0, 1])

    fpr, tpr, _ = roc_curve(true_bin, pred_bin)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

