import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from CONSTANTS import char2idx, idx2char, max_smile_len, device
from smiles_utilis import get_validity_metric, similarity_score
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
            
            selected_log_probs = torch.zeros(batch_size, device=device)
            selected_log_probs[active_indices] = probs[active_indices, next_tokens[active_indices]]
            log_probs_list.append(selected_log_probs)
            
            if not active_mask.any():
                break
        return generated, log_probs_list

    
    def monte_carlo_search(self, discriminator, partial_seqs, max_len,fine_tune_flag=False,inhibitors=None, num_samples=20, temperature=1.0):
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
                probs = probs / probs_sum 
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
            # Compute rewards for the batch
            if fine_tune_flag:
                batch_rewards=self.fine_tune_reward(current_seqs,inhibitors )
            else:
                batch_rewards = self.compute_reward(current_seqs, discriminator)  # [batch_size]
                
            rewards[:, sample_idx] = batch_rewards

        # Average rewards across samples for each sequence
        return rewards.mean(dim=1)  # [batch_size]

    def compute_reward(self, sequences, discriminator):
        batch_size = len(sequences)
        rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)

        # Convert sequences to SMILES strings
        smiles_list = [''.join([idx2char[t] for t in seq]) for seq in sequences]
        # Compute validity scores
        validity_scores = torch.tensor([get_validity_metric(smiles) for smiles in smiles_list], dtype=torch.float32, device=device)

        # Pad sequences for discriminator
        seq_tensor = torch.full((batch_size, max_smile_len), char2idx["<PAD>"], dtype=torch.long, device=device)
        
        for i, seq in enumerate(sequences):
            seq_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        # Compute discriminator scores
        with torch.no_grad():
            disc_scores = discriminator(seq_tensor)  # [batch_size, 1]
            disc_scores = torch.sigmoid(disc_scores).squeeze(-1)  # [batch_size]

        # Compute combined rewards
        rewards = 0.5 * validity_scores + 0.5 * disc_scores
        return rewards

    
    def fine_tune_reward(self, sequences,inhibitors ):
        batch_size = len(sequences)
        rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)
        # Convert sequences to SMILES strings
        smiles_list = [''.join([idx2char[t] for t in seq]) for seq in sequences]
        # Compute validity scores
        validity_scores = torch.tensor([get_validity_metric(smiles) for smiles in smiles_list], dtype=torch.float32, device=device)
        sim_scores = similarity_score(smiles_list, inhibitors)
        sim_scores = torch.tensor(sim_scores, dtype=torch.float32, device=device)
        rewards = 0.5 * validity_scores  + 0.5 * sim_scores
        return rewards


    def train_step(self, start_token, max_len, optimizer, discriminator, scaler,fine_tune_flag=False,inhibitors=None, temperature=1.0, num_mc_samples=20, batch_size=16):
        self.train()
        optimizer.zero_grad()

        with autocast():
            # Generate a batch of sequences
            generated_batch, log_probs_list_batch = self.sample(start_token, max_len, batch_size=batch_size, temperature=temperature)
            loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
            entropy_loss = 0.0

            # Convert generated batch to list of sequences
            generated_seqs = [generated_batch[i].tolist() for i in range(batch_size)]

            # Accumulate policy gradient loss
            policy_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)

            for t in range(len(log_probs_list_batch)):
                # Get partial sequences up to step t+1
                partial_seqs = [seq[:t+1] for seq in generated_seqs]
                q_values = self.monte_carlo_search(discriminator, partial_seqs, max_len,fine_tune_flag,inhibitors, num_mc_samples, temperature)  # [batch_size]
                
                # Normalize Q-values to reduce variance
                q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-8)
                
                # Compute policy gradient loss for this timestep
                log_probs = log_probs_list_batch[t]  # [batch_size]
                policy_loss = policy_loss - (log_probs * q_values).mean()  # Accumulate loss

                # Add entropy regularization
                probs = torch.exp(log_probs)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                entropy_loss = entropy_loss + entropy

            if len(log_probs_list_batch) > 0:
                # Combine policy loss and entropy loss
                loss = (policy_loss + 0.1 * entropy_loss) / len(log_probs_list_batch)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        return loss.item()