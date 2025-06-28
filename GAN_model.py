import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from GAN_utils import get_validity_metric

max_smile_len=128
min_smile_len=32
smiles_characters = [
    'Cl', 'Br','Na','Si','Mg','Zn','Se','Te','se','As','te','Ag','Al',
    'B','C', 'N', 'O', 'P', 'S', 'F', 'I','K',
    'b', 'c', 'n', 'o', 'p', 's',
    '-', '=', '#', ':', '/', '\\',
    '(', ')',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '[', ']',
    '+', '-', '@', '@@', '.', '%',
    'H', 'h'  ,
    '<EOS>',
    '<SOS>',
    '<PAD>'
]

char2idx = {ch: i for i, ch in enumerate(smiles_characters)}
idx2char = {i: ch for i, ch in enumerate(smiles_characters)}
vocab_size = len(smiles_characters)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##real->1  && fake->0
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, conv_filters=3, kernel_size=3, sequence_length=max_smile_len):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, conv_filters, 5, padding=2)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters * 2, 7, padding=3)
        self.relu = nn.ReLU()
        # Compute flattened size
        self.flattened_size = (conv_filters * 2) * sequence_length
        self.fc = nn.Linear(self.flattened_size, 1)
        self.dropout = nn.Dropout(0.3)  # Dropout layer
    
    def forward(self, x):
        x = self.embedding(x)                     # (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)                    # (batch_size, embedding_dim, sequence_length)
        x = self.relu(self.conv1(x))              # (batch_size, conv_filters, sequence_length)
        x = self.relu(self.conv2(x))              # (batch_size, conv_filters*2, sequence_length)
        x = self.dropout(x)                       # Dropout after conv2
        x = x.view(x.size(0), -1)                 # Flatten
        x = self.fc(x)                            # (batch_size, 1)
        return x

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=256, num_layers=4):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.linear(output)
        log_probs = self.softmax(logits)
        return log_probs, hidden

    def sample(self, start_token, max_len, batch_size=1, temperature=1.0):
        input_token = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        hidden = None
        generated = torch.full((batch_size, max_len), char2idx["<PAD>"], dtype=torch.long, device=device)
        generated[:, 0] = start_token
        log_probs_list = []
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)


        for t in range(max_len - 1):
            log_probs, hidden = self.forward(input_token, hidden)
            log_probs = log_probs[:, -1, :]  # [batch_size, vocab_size]
            logits = log_probs / temperature
            logits[:, char2idx["<PAD>"]] = -float('inf')
            logits[:, char2idx["<SOS>"]] = -float('inf')
            probs = torch.exp(logits)
            probs_sum = probs.sum(dim=-1, keepdim=True)
           
            probs = probs / (probs_sum + 1e-10)
            
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            
            active_indices = active_mask.nonzero(as_tuple=True)[0]
           
            input_token = torch.full((batch_size, 1), char2idx["<PAD>"], dtype=torch.long, device=device)
            input_token[active_indices] = next_tokens[active_indices].unsqueeze(1)
            generated[active_indices, t + 1] = next_tokens[active_indices]
            
            eos_mask = (next_tokens == char2idx["<EOS>"]) & active_mask
            active_mask[eos_mask] = False
            
            try:
                selected_log_probs = torch.zeros(batch_size, device=device)
                selected_log_probs[active_indices] = probs[active_indices, next_tokens[active_indices]]
                log_probs_list.append(selected_log_probs)
            except IndexError as e:
                print(f"IndexError in log_probs indexing at step {t}: {e}")
                break
            
            if not active_mask.any():
                break
        return generated, log_probs_list

    
    def sample(self, start_token, max_len, batch_size=1, temperature=1.0):
        input_token = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        hidden = None
        generated = torch.full((batch_size, max_len), char2idx["<PAD>"], dtype=torch.long, device=device)
        generated[:, 0] = start_token
        log_probs_list = []
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)


        for t in range(max_len - 1):
            log_probs, hidden = self.forward(input_token, hidden)
            log_probs = log_probs[:, -1, :]  # [batch_size, vocab_size]
            logits = log_probs / temperature
            logits[:, char2idx["<PAD>"]] = -float('inf')
            logits[:, char2idx["<SOS>"]] = -float('inf')
            probs = torch.exp(logits)
            probs_sum = probs.sum(dim=-1, keepdim=True)
           
            probs = probs / (probs_sum + 1e-10)
            
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            
            active_indices = active_mask.nonzero(as_tuple=True)[0]
           
            input_token = torch.full((batch_size, 1), char2idx["<PAD>"], dtype=torch.long, device=device)
            input_token[active_indices] = next_tokens[active_indices].unsqueeze(1)
            generated[active_indices, t + 1] = next_tokens[active_indices]
            
            eos_mask = (next_tokens == char2idx["<EOS>"]) & active_mask
            active_mask[eos_mask] = False
            
            try:
                selected_log_probs = torch.zeros(batch_size, device=device)
                selected_log_probs[active_indices] = log_probs[active_indices, next_tokens[active_indices]]
                log_probs_list.append(selected_log_probs)
            except IndexError as e:
                print(f"IndexError in log_probs indexing at step {t}: {e}")
                break
            
            if not active_mask.any():
                break
        return generated, log_probs_list

    
    def monte_carlo_search(self, discriminator, partial_seqs, max_len, num_samples=20, temperature=1.0):
        batch_size = len(partial_seqs)
        rewards = torch.zeros(batch_size, num_samples, dtype=torch.float32, device=device)

        for sample_idx in range(num_samples):
            current_seqs = [seq.copy() for seq in partial_seqs]  # Copy partial sequences
            input_tokens = torch.tensor([[seq[-1]] for seq in current_seqs], dtype=torch.long, device=device)  # [batch_size, 1]
            hidden = None

            # Initialize hidden state with partial sequences
            if len(current_seqs[0]) > 1:
                seq_tensor = torch.tensor([seq[:-1] for seq in current_seqs], dtype=torch.long, device=device)  # [batch_size, seq_len-1]
                _, hidden = self.forward(seq_tensor, hidden)

            active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            for _ in range(max_len - len(current_seqs[0])):
                log_probs, hidden = self.forward(input_tokens, hidden)  # [batch_size, 1, vocab_size]
                logits = log_probs[:, -1, :].clone()  # [batch_size, vocab_size]
                logits[:, char2idx["<PAD>"]] = -float('inf')
                logits[:, char2idx["<SOS>"]] = -float('inf')
                probs = torch.exp(logits / temperature)
                probs_sum = probs.sum(dim=-1, keepdim=True)
                probs = probs / (probs_sum + 1e-10)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch_size]

                active_indices = active_mask.nonzero(as_tuple=True)[0]
                input_tokens = torch.full((batch_size, 1), char2idx["<PAD>"], dtype=torch.long, device=device)
                input_tokens[active_indices] = next_tokens[active_indices].unsqueeze(1)
                for idx in active_indices:
                    current_seqs[idx.item()].append(next_tokens[idx].item())
                eos_mask = (next_tokens == char2idx["<EOS>"]) & active_mask
                active_mask[eos_mask] = False
                if not active_mask.any():
                    break

         
            batch_rewards = self.compute_reward(current_seqs, discriminator)  # [batch_size]
            rewards[:, sample_idx] = batch_rewards

        # Average rewards across samples for each sequence
        return rewards.mean(dim=1)  # [batch_size]

    def compute_reward(self, sequences, discriminator):
        batch_size = len(sequences)
        rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)


        smiles_list = [''.join([idx2char[t] for t in seq]) for seq in sequences]


        validity_scores = torch.tensor([get_validity_metric(smiles) for smiles in smiles_list], dtype=torch.float32, device=device)

        # Pad sequences for discriminator
        max_seq_len = max(len(seq) for seq in sequences)
        seq_tensor = torch.full((batch_size, max_smile_len), char2idx["<PAD>"], dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            seq_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

   
        with torch.no_grad():
            disc_scores = discriminator(seq_tensor)  # [batch_size, 1]
            disc_scores = torch.sigmoid(disc_scores).squeeze(-1)  # [batch_size]


        rewards = 0.5 * validity_scores + 0.5 * disc_scores
            
        
        return rewards

    def train_step(self, start_token, max_len, optimizer, discriminator, scaler, temperature=1.0, num_mc_samples=20, batch_size=16):
        self.train()
        optimizer.zero_grad()

        with autocast():
            
            generated_batch, log_probs_list_batch = self.sample(start_token, max_len, batch_size=batch_size, temperature=temperature)
            loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
            entropy_loss = 0.0

         
            generated_seqs = [generated_batch[i].tolist() for i in range(batch_size)]

            policy_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)

            for t in range(len(log_probs_list_batch)):
                # Get partial sequences up to step t+1
                partial_seqs = [seq[:t+1] for seq in generated_seqs]
                q_values = self.monte_carlo_search(discriminator, partial_seqs, max_len, num_mc_samples, temperature)  # [batch_size]
                
                # Normalize Q-values 
                q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-8)
                
                log_probs = log_probs_list_batch[t]  # [batch_size]
                policy_loss = policy_loss - (log_probs * q_values).mean()  # Accumulate loss

                # Add entropy regularization
                probs = torch.exp(log_probs)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                entropy_loss = entropy_loss + entropy

            if len(log_probs_list_batch) > 0:
                loss = (policy_loss + 0.1 * entropy_loss) / len(log_probs_list_batch)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        return loss.item()