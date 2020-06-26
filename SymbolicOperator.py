import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from PositionalEncoding import PositionalEncoding
from Attention import Attention


class SymbolicOperator(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size):
        super().__init__()

        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.max_len = 50
        self.actions_dim = 200
        self.scratch_keys_dim = 128
        self.scratch_values_dim = self.actions_dim
        self.program_dim = 200
        self.n_pointers = 3

        self.scratch_keys = PositionalEncoding(self.scratch_keys_dim, max_len=self.max_len).pe[:, 0, :]
        self.initial_scratch_value = nn.Parameter(torch.randn(self.actions_dim))
        self.attention = Attention()
        self.program_embedding = nn.Embedding(self.in_vocab_size, self.program_dim)
        self.primitive_embedding = nn.Embedding(self.in_vocab_size, self.actions_dim)

        self.gate_linear = nn.Linear(self.scratch_keys_dim, 1)
        self.executor_rnn_cell = nn.GRUCell(input_size=1, hidden_size=self.scratch_keys_dim * (self.n_pointers + 1))
        self.out_linear = nn.Linear(self.actions_dim, self.out_vocab_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def init_executor_hidden(self, batch_size):
        return self.scratch_keys[0].repeat(batch_size, self.n_pointers + 1)

    def forward(self, instructions, true_actions):
        batch_size = len(instructions)

        # Pad sequences of true actions
        seq_lens = [a.shape[0] for a in true_actions]
        padded_true_actions = pad_sequence(true_actions, padding_value=-100)

        # Pad sequences in instructions
        seq_lens = [ins.shape[0] for ins in instructions]
        instructions = pad_sequence(instructions)

        # Remove <SOS> token from true actions
        padded_true_actions = padded_true_actions[1:, :]
        max_len = padded_true_actions.shape[0]
        max_program_steps = max_len

        executor_hidden = self.init_executor_hidden(batch_size)
        scratch_keys = self.scratch_keys[:max_len, :].expand(batch_size, max_len, self.scratch_keys_dim)
        scratch_values = self.initial_scratch_value.repeat(batch_size, max_len, 1)

        for word_idx in instructions:
            program = self.program_embedding(word_idx)
            primitive = self.primitive_embedding(word_idx)
            for step in range(max_program_steps):
                init_pointer = executor_hidden[:, 0:self.scratch_keys_dim]
                read_pointer = executor_hidden[:, self.scratch_keys_dim:2*self.scratch_keys_dim]
                write_pointer = executor_hidden[:, 2*self.scratch_keys_dim:3*self.scratch_keys_dim]
                gate = torch.sigmoid(self.gate_linear(executor_hidden[:, 3*self.scratch_keys_dim:4*self.scratch_keys_dim]))

                read_value, read_attn = self.attention(read_pointer.unsqueeze(1), scratch_keys, scratch_values)
                new_value = gate * primitive + (1 - gate) * read_value.squeeze(1)
                write_mask = F.softmax(torch.bmm(write_pointer.unsqueeze(1), scratch_keys.transpose(1, 2)), dim=-1)

                write_mask_flatten = write_mask.view(batch_size * max_len, 1).unsqueeze(2)
                new_value_flatten = new_value.unsqueeze(1).expand(batch_size, max_len, self.scratch_values_dim).reshape(batch_size * max_len, -1).unsqueeze(1)
                new_value_weighted = torch.bmm(write_mask_flatten, new_value_flatten)
                scratch_values_flatten = scratch_values.view(batch_size * max_len, -1).unsqueeze(1)
                previous_values_weighted = torch.bmm(1 - write_mask_flatten, scratch_values_flatten)
                scratch_values = (new_value_weighted + previous_values_weighted).view(batch_size, max_len, -1)

                executor_hidden = self.executor_rnn_cell(torch.zeros(batch_size, 1).to(self.device), executor_hidden)

        actions = F.log_softmax(self.out_linear(scratch_values), dim=-1)

        # Prepare decoder outputs and actions for NLLLoss (ignore -100)
        actions = actions.transpose(1, 2)

        padded_true_actions = padded_true_actions.permute(1, 0) # (batch, seq_len)

        return actions, padded_true_actions
