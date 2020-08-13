import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from PositionalEncoding import PositionalEncoding
from Attention import Attention
from modules.attention_activation import AttentionActivation


class SymbolicOperator(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, eos_idx,
        max_program_steps=3, scratch_max_len=50,
        pretrained_gate=None,
        gate_activation_train='gumbel_st', gate_activation_eval='argmax',
        gate_activation_temperature=1.0,
        read_activation_train='softmax', read_activation_eval='softmax',
        write_activation_train='softmax', write_activation_eval='softmax',
        use_adaptive_steps=False, keep_going_input='read_value',
        normalize_act_by_input=False,
        separate_primitive_keep_going=False,
        include_init_pointer=True,
    ):
        super().__init__()

        self.separate_primitive_keep_going = separate_primitive_keep_going
        self.include_init_pointer = include_init_pointer

        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.scratch_max_len = scratch_max_len
        self.max_program_steps = max_program_steps
        self.actions_dim = 200
        self.scratch_keys_dim = 128
        self.scratch_values_dim = self.out_vocab_size
        self.program_dim = 200
        n_pointers = 2
        if self.include_init_pointer:
            n_pointers += 1
        self.n_pointers = n_pointers
        self.executor_hidden_dim = self.scratch_keys_dim * self.n_pointers

        scratch_keys = PositionalEncoding(self.scratch_keys_dim, max_len=self.scratch_max_len).pe[:, 0, :]
        self.register_buffer('scratch_keys', scratch_keys)
        self.initial_scratch_value = nn.Parameter(torch.zeros(self.scratch_values_dim).scatter_(0, torch.tensor([eos_idx]), 1), requires_grad=False)
        self.gate_attention_activation = AttentionActivation(
            sample_train=gate_activation_train,
            sample_infer=gate_activation_eval,
            initial_temperature=gate_activation_temperature,
        )
        self.read_attention_activation = AttentionActivation(
            sample_train=read_activation_train,
            sample_infer=read_activation_eval,
            initial_temperature=1.,
        )
        self.write_attention_activation = AttentionActivation(
            sample_train=write_activation_train,
            sample_infer=write_activation_eval,
            initial_temperature=1.,
        )
        self.read_attention = Attention(attention_activation=self.read_attention_activation)
        self.pretrained_gate = pretrained_gate
        if self.pretrained_gate is not None:
            self.gate_embedding = nn.Embedding.from_pretrained(self.pretrained_gate, freeze=True)
        else:
            self.gate_embedding = nn.Embedding(self.in_vocab_size, 2)
        self.program_embedding = nn.Embedding(self.in_vocab_size, self.program_dim)
        self.primitive_embedding = nn.Embedding(self.in_vocab_size, self.scratch_values_dim)

        self.use_adaptive_steps = use_adaptive_steps
        self.normalize_act_by_input = normalize_act_by_input
        self.initial_keep_going_gate = nn.Parameter(torch.tensor([1], dtype=torch.float), requires_grad=False)
        self.initial_keep_going_loss = nn.Parameter(torch.tensor([0], dtype=torch.float), requires_grad=False)
        self.keep_going_input = keep_going_input
        self.keep_going_hidden_dim = self.scratch_values_dim
        self.initial_keep_going_hidden = nn.Parameter(torch.randn([self.keep_going_hidden_dim], dtype=torch.float))
        if self.keep_going_input == 'read_value':
            keep_going_input_dim = self.scratch_values_dim
        elif self.keep_going_input == 'executor_hidden':
            keep_going_input_dim = self.executor_hidden_dim
        elif self.keep_going_input == 'read_value+executor_hidden':
            keep_going_input_dim = self.scratch_values_dim + self.executor_hidden_dim
        elif self.keep_going_input == 'other_hidden':
            keep_going_input_dim = self.keep_going_hidden_dim
            self.executor_hidden_dim += self.keep_going_hidden_dim
        else:
            raise ValueError('Invalid keep_going_input: %s' % (self.keep_going_input))
        self.keep_going_linear = nn.Linear(keep_going_input_dim, 1)

        self.executor_rnn_cell = nn.GRUCell(input_size=self.program_dim, hidden_size=self.executor_hidden_dim)

        self.scratch_history = []

    @property
    def device(self):
        return next(self.parameters()).device

    def init_executor_hidden(self, batch_size):
        executor_hidden = self.scratch_keys[0].repeat(batch_size, self.n_pointers)
        if self.keep_going_input == 'other_hidden':
            executor_hidden = torch.cat([executor_hidden, self.initial_keep_going_hidden.repeat(batch_size, 1)], dim=-1)

        return executor_hidden

    def set_extra_config(self, extra_config):
        if 'gate_activation_train' in extra_config:
            self.gate_attention_activation.sample_train = extra_config['gate_activation_train']
        if 'gate_activation_eval' in extra_config:
            self.gate_attention_activation.sample_infer = extra_config['gate_activation_eval']
        if 'gate_activation_temperature' in extra_config:
            new_temperature = torch.tensor(
                extra_config['gate_activation_temperature'],
                requires_grad=False, device=self.device)
            self.gate_attention_activation.temperature = new_temperature
        if 'read_activation_train' in extra_config:
            self.read_attention_activation.sample_train = extra_config['read_activation_train']
        if 'read_activation_train' in extra_config:
            self.read_attention_activation.sample_infer = extra_config['read_activation_train']
        if 'write_activation_train' in extra_config:
            self.write_attention_activation.sample_train = extra_config['write_activation_train']
        if 'write_activation_train' in extra_config:
            self.write_attention_activation.sample_infer = extra_config['write_activation_train']

    def forward(self, instructions, true_actions, extra_config=None):
        if extra_config:
            self.set_extra_config(extra_config)

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

        total_keep_going_loss = self.initial_keep_going_loss.repeat(batch_size, 1)
        for word_idx in instructions:
            if not self.training:
                self.scratch_history.append([])
            gate = self.gate_embedding(word_idx).unsqueeze(1)
            if self.pretrained_gate is None:
                gate = self.gate_attention_activation(
                    gate,
                    torch.zeros_like(gate, dtype=torch.bool).to(self.device),
                    torch.zeros(batch_size, 1, 2).to(self.device),
                )
            program = self.program_embedding(word_idx)
            primitive = self.primitive_embedding(word_idx)
            keep_going_gate = self.initial_keep_going_gate
            cur_keep_going_loss = 0
            for step in range(max_program_steps):
                read_pointer = executor_hidden[:, 0:self.scratch_keys_dim]
                write_pointer = executor_hidden[:, self.scratch_keys_dim:2*self.scratch_keys_dim]

                read_value, read_attn = self.read_attention(read_pointer.unsqueeze(1), scratch_keys, scratch_values)
                read_value = read_value.squeeze(1)
                new_value = torch.bmm(gate, torch.stack([primitive, read_value]).transpose(0, 1)).squeeze(1)
                write_mask = torch.bmm(write_pointer.unsqueeze(1), scratch_keys.transpose(1, 2))
                write_mask = self.write_attention_activation(
                    write_mask,
                    torch.zeros_like(write_mask, dtype=torch.bool).to(self.device),
                    torch.zeros(batch_size, 1, self.scratch_keys_dim).to(self.device),
                )

                if self.use_adaptive_steps:
                    if self.keep_going_input == 'read_value':
                        keep_going_input = read_value
                    elif self.keep_going_input == 'executor_hidden':
                        keep_going_input = executor_hidden
                    elif self.keep_going_input == 'read_value+executor_hidden':
                        keep_going_input = torch.cat([read_value, executor_hidden], dim=-1)
                    elif self.keep_going_input == 'other_hidden':
                        keep_going_input = executor_hidden[:, self.n_pointers*self.scratch_keys_dim:]
                    keep_going_prob = torch.sigmoid(self.keep_going_linear(keep_going_input))
                    cur_keep_going_loss += keep_going_prob
                    operator_keep_going_gate = keep_going_gate * keep_going_prob
                    if self.separate_primitive_keep_going:
                        primitive_keep_going_gate = torch.ones_like(
                            operator_keep_going_gate,
                            dtype=torch.float,
                            device=self.device
                        )
                        keep_going_gate = torch.bmm(gate, torch.stack([
                            primitive_keep_going_gate,
                            operator_keep_going_gate
                        ]).transpose(0, 1)).squeeze(1)
                    else:
                        keep_going_gate = operator_keep_going_gate

                    write_mask = torch.bmm(
                        keep_going_gate.unsqueeze(1),
                        write_mask,
                    )

                write_mask_flatten = write_mask.view(batch_size * max_len, 1).unsqueeze(2)
                new_value_flatten = new_value.unsqueeze(1).expand(batch_size, max_len, self.scratch_values_dim).reshape(batch_size * max_len, -1).unsqueeze(1)
                new_value_weighted = torch.bmm(write_mask_flatten, new_value_flatten)
                scratch_values_flatten = scratch_values.view(batch_size * max_len, -1).unsqueeze(1)
                previous_values_weighted = torch.bmm(1 - write_mask_flatten, scratch_values_flatten)
                scratch_values = (new_value_weighted + previous_values_weighted).view(batch_size, max_len, -1)

                executor_hidden = torch.bmm(gate, torch.stack([
                    executor_hidden,
                    self.executor_rnn_cell(program, executor_hidden),
                ]).transpose(0, 1)).squeeze(1)

                if not self.training:
                    self.scratch_history[-1].append([
                        torch.round(gate),
                        torch.round(keep_going_gate),
                        torch.round(read_attn),
                        torch.round(write_mask),
                        scratch_values.cpu().detach()
                    ])
            total_keep_going_loss += cur_keep_going_loss / max_program_steps
        if self.normalize_act_by_input:
            total_keep_going_loss /= len(instructions)

        actions = F.log_softmax(scratch_values, dim=-1)

        # Prepare decoder outputs and actions for NLLLoss (ignore -100)
        actions = actions.transpose(1, 2)

        padded_true_actions = padded_true_actions.permute(1, 0) # (batch, seq_len)

        if self.use_adaptive_steps:
            return actions, padded_true_actions, total_keep_going_loss
        else:
            return actions, padded_true_actions
