import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from PositionalEncoding import PositionalEncoding
from Attention import Attention
from modules.attention_activation import AttentionActivation


class SymbolicOperator(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, eos_idx,
        max_program_steps=3, gate_activation='gumbel_st',
    ):
        super().__init__()

        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.max_len = 50
        self.max_program_steps = max_program_steps
        self.actions_dim = 200
        self.scratch_keys_dim = 128
        self.scratch_values_dim = self.out_vocab_size
        self.program_dim = 200
        self.n_pointers = 3

        scratch_keys = PositionalEncoding(self.scratch_keys_dim, max_len=self.max_len).pe[:, 0, :]
        self.register_buffer('scratch_keys', scratch_keys)
        self.initial_scratch_value = nn.Parameter(torch.zeros(self.scratch_values_dim).scatter_(0, torch.tensor([eos_idx]), 1), requires_grad=False)
        if gate_activation == 'gumbel_st':
            gate_sample_train = 'gumbel_st'
            gate_sample_infer = 'argmax'
        elif gate_activation == 'softmax':
            gate_sample_train = 'softmax'
            gate_sample_infer = 'softmax'
        self.gate_attention_activation = AttentionActivation(
            sample_train=gate_sample_train,
            sample_infer=gate_sample_infer,
            initial_temperature=1.,
        )
        self.read_attention_activation = AttentionActivation(
            sample_train='softmax',
            sample_infer='softmax',
            initial_temperature=1.,
        )
        self.write_attention_activation = AttentionActivation(
            sample_train='softmax',
            sample_infer='softmax',
            initial_temperature=1.,
        )
        self.read_attention = Attention(attention_activation=self.read_attention_activation)
        self.gate_embedding = nn.Embedding(self.in_vocab_size, 2)
        self.program_embedding = nn.Embedding(self.in_vocab_size, self.program_dim)
        self.primitive_embedding = nn.Embedding(self.in_vocab_size, self.scratch_values_dim)

        self.executor_rnn_cell = nn.GRUCell(input_size=self.program_dim, hidden_size=self.scratch_keys_dim * self.n_pointers)

        self.scratch_history = []

    @property
    def device(self):
        return next(self.parameters()).device

    def init_executor_hidden(self, batch_size):
        return self.scratch_keys[0].repeat(batch_size, self.n_pointers)

    def forward(self, instructions, true_actions):
        self.scratch_history = []
        batch_size = len(instructions)

        # Pad sequences of true actions
        seq_lens = [a.shape[0] for a in true_actions]
        padded_true_actions = pad_sequence(true_actions, padding_value=-100)

        # Pad sequences in instructions
        seq_lens = [ins.shape[0] for ins in instructions]
        instructions = pad_sequence(instructions)
        # Remove <SOS> and <EOS> tokens from instructions
        instructions = instructions[1:-1, :]

        # Remove <SOS> token from true actions
        padded_true_actions = padded_true_actions[1:, :]
        max_len = padded_true_actions.shape[0]
        max_program_steps = self.max_program_steps

        executor_hidden = self.init_executor_hidden(batch_size)
        scratch_keys = self.scratch_keys[:max_len, :].expand(batch_size, max_len, self.scratch_keys_dim)
        scratch_values = self.initial_scratch_value.repeat(batch_size, max_len, 1)

        for word_idx in instructions:
            if not self.training:
                self.scratch_history.append([])
            gate = self.gate_embedding(word_idx).unsqueeze(1)
            gate = self.gate_attention_activation(
                gate,
                torch.zeros_like(gate, dtype=torch.bool).to(self.device),
                torch.zeros(batch_size, 1, 2).to(self.device),
            )
            program = self.program_embedding(word_idx)
            primitive = self.primitive_embedding(word_idx)
            for step in range(max_program_steps):
                init_pointer = executor_hidden[:, 0:self.scratch_keys_dim]
                read_pointer = executor_hidden[:, self.scratch_keys_dim:2*self.scratch_keys_dim]
                write_pointer = executor_hidden[:, 2*self.scratch_keys_dim:3*self.scratch_keys_dim]

                read_value, read_attn = self.read_attention(read_pointer.unsqueeze(1), scratch_keys, scratch_values)
                read_value = read_value.squeeze(1)
                new_value = torch.bmm(gate, torch.stack([primitive, read_value]).transpose(0, 1)).squeeze(1)
                write_mask = torch.bmm(write_pointer.unsqueeze(1), scratch_keys.transpose(1, 2))
                write_mask = self.write_attention_activation(
                    write_mask,
                    torch.zeros_like(write_mask, dtype=torch.bool).to(self.device),
                    torch.zeros(batch_size, 1, self.scratch_keys_dim).to(self.device),
                )

                write_mask_flatten = write_mask.view(batch_size * max_len, 1).unsqueeze(2)
                new_value_flatten = new_value.unsqueeze(1).expand(batch_size, max_len, self.scratch_values_dim).reshape(batch_size * max_len, -1).unsqueeze(1)
                new_value_weighted = torch.bmm(write_mask_flatten, new_value_flatten)
                scratch_values_flatten = scratch_values.view(batch_size * max_len, -1).unsqueeze(1)
                previous_values_weighted = torch.bmm(1 - write_mask_flatten, scratch_values_flatten)
                scratch_values = (new_value_weighted + previous_values_weighted).view(batch_size, max_len, -1)

                executor_hidden = self.executor_rnn_cell(program, executor_hidden)

                if not self.training:
                    self.scratch_history[-1].append([
                        torch.round(gate),
                        torch.round(read_attn),
                        torch.round(write_mask),
                        scratch_values.cpu().detach()
                    ])

        actions = F.log_softmax(scratch_values, dim=-1)

        # Prepare decoder outputs and actions for NLLLoss (ignore -100)
        actions = actions.transpose(1, 2)

        padded_true_actions = padded_true_actions.permute(1, 0) # (batch, seq_len)

        return actions, padded_true_actions
