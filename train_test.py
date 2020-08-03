from comet_ml import Experiment

import os
import argparse
import json
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from torch.utils.data import DataLoader
from data import ScanDataset,ScanAugmentedDataset,MTDataset,SCAN_collate
from SyntacticAttention import *
from SymbolicOperator import SymbolicOperator
from utils import *

comet_args = {
    'project_name': os.environ.get('COMET_PROJECT_NAME'),
    'workspace': 'andresespinosapc',
}
if os.environ.get('COMET_DISABLE'):
    comet_args['disabled'] = True
    comet_args['api_key'] = ''
if os.environ.get('COMET_OFFLINE'):
    comet_args['api_key'] = ''
    comet_args['offline_directory'] = 'comet_offline'
    experiment = OfflineExperiment(**comet_args)
else:
    experiment = Experiment(**comet_args)

def log_comet_parameters(opt):
    opt_dict = vars(opt)
    for key in opt_dict.keys():
        experiment.log_parameter(key, opt_dict[key])

def validate_args(parser, args):
    if args.exp_name is None and not comet_args.get('disabled'):
        parser.error('Please provide exp_name if logging to CometML')


parser = argparse.ArgumentParser()

parser.add_argument('--extra_config_path', type=str, default=None, help='Path to YAML file with extra config')
parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for CometML logging')
parser.add_argument('--use_scan_augmented', type=str2bool, default=False, help='Use ScanAugmentedDataset')
parser.add_argument('--model', type=str, choices=['syntactic_attention', 'symbolic_operator'], default='syntactic_attention')
parser.add_argument('--auto_val_split', type=str2bool, default=True)
parser.add_argument('--save_all_checkpoints', type=str2bool, default=False)
parser.add_argument('--max_program_steps', type=int, default=3, help="Maximum program steps for symbolic operator")
parser.add_argument('--max_output_len', type=int, default=50, help='Maximum output length. Examples with less than this will be removed')
parser.add_argument('--fixed_gate', type=str2bool, default=False, help='Use fixed gate depending on primitives and operators')
parser.add_argument('--gate_activation_train', type=str, choices=['gumbel_st', 'softmax', 'argmax', 'softmax_st'], default='gumbel_st', help='Activation for gate of symbolic operator')
parser.add_argument('--gate_activation_eval', type=str, choices=['argmax', 'softmax'], default='argmax', help='Activation for gate of symbolic operator')
parser.add_argument('--gate_activation_temperature', type=float, default=1.0, help='Temperature for gumbel or softmax_st activations')
parser.add_argument('--read_activation_train', type=str, choices=['gumbel_st', 'softmax'], default='softmax', help='Activation for read of symbolic operator')
parser.add_argument('--read_activation_eval', type=str, choices=['argmax', 'softmax'], default='softmax', help='Activation for read of symbolic operator')
parser.add_argument('--write_activation_train', type=str, choices=['gumbel_st', 'softmax'], default='softmax', help='Activation for write of symbolic operator')
parser.add_argument('--write_activation_eval', type=str, choices=['argmax', 'softmax'], default='softmax', help='Activation for write of symbolic operator')
parser.add_argument('--use_adaptive_steps', type=str2bool, default=False, help='Use an adaptive number of reasoning steps per word')
parser.add_argument('--adaptive_steps_loss_weight', type=float, default=1.0, help='Weight to ponder adaptive steps loss')
parser.add_argument('--keep_going_input', type=str, choices=['read_value', 'executor_hidden', 'read_value+executor_hidden'], default='read_value', help='Weight to ponder adaptive steps loss')

# Data
parser.add_argument('--dataset', choices=['SCAN','MT'],
                    default='SCAN',
                    help='Dataset class to use')
parser.add_argument('--flip', type=str2bool, default=False,
                    help='Flip source and target for MT dataset')
parser.add_argument('--train_data_file',
                    default='data/SCAN/tasks_train_addprim_jump.txt',
                    help='Path to training set')
parser.add_argument('--val_data_file',
                    default='data/SCAN/tasks_test_addprim_jump.txt',
                    help='Path to validation set')
parser.add_argument('--test_data_file',
                    default='data/SCAN/tasks_test_addprim_jump.txt',
                    help='Path to test set')
parser.add_argument('--load_vocab_json',default='vocab.json',
                    help='Path to vocab json file')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Samples per batch')
parser.add_argument('--num_iters', type=int, default=200000,
                    help='Number of optimizer steps before stopping')
parser.add_argument('--seed', type=int, default=1,
                    help='Seed for random number generators')

# Model hyperparameters
parser.add_argument('--rnn_type', choices=['GRU', 'LSTM'],
                    default='LSTM', help='Type of rnn to use.')
parser.add_argument('--m_hidden_dim', type=int, default=120,
                    help='Number of hidden units in semantic embeddings')
parser.add_argument('--x_hidden_dim', type=int, default=200,
                    help='Number of hidden units in syntax rnn')
parser.add_argument('--enc_n_layers', type=int, default=2,
                    help='Number of layers in encoder RNN')
parser.add_argument('--dec_n_layers', type=int, default=1,
                    help='Number of layers in decoder RNN')
parser.add_argument('--dropout_p', type=float, default=0.5,
                    help='Dropout rate')
parser.add_argument('--seq_sem', type=str2bool, default=False,
                    help='Semantic embeddings also processed with RNN.')
parser.add_argument('--syn_act', type=str2bool, default=False,
                    help='Syntactic information also used for action')
parser.add_argument('--sem_mlp', type=str2bool, default=False,
                    help='Nonlinear semantic layer with ReLU')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights')

# Optimization
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Fixed learning rate for Adam optimizer')
parser.add_argument('--clip_norm', type=float, default=5.0,
                    help='Maximum L2-norm at which gradients will be clipped.')

# Output options
parser.add_argument('--results_dir', default='results',
                    help='Results subdirectory to save results')
parser.add_argument('--out_data_file', default='results.json',
                    help='Name of output data file')
parser.add_argument('--checkpoint_dir',default=None,
                    help='Path to output saved weights.')
parser.add_argument('--checkpoint_every', type=int, default=5,
                    help='Epochs before evaluating model and saving weights')
parser.add_argument('--record_loss_every', type=int, default=400,
                    help='iters before printing and recording loss')

PRIMITIVES = [
    'jump', 'right', 'thrice', 'run', 'left', 'walk', 'look',
]
OPERATORS = [
    'opposite', 'twice', 'and', 'thrice', 'around', 'after',
    'turn', '<SOS>', '<EOS>', '<NULL>',
]

def get_pretrained_gate(vocab):
    pretrained_gate = []
    for i in range(len(vocab['in_idx_to_token'])):
        token = vocab['in_idx_to_token'][str(i)]
        if token in PRIMITIVES:
            pretrained_gate.append([1, 0])
        elif token in OPERATORS:
            pretrained_gate.append([0, 1])
        else:
            raise ValueError('Invalid token', token)

    return torch.tensor(pretrained_gate, dtype=torch.float)

def load_dataloaders(args, vocab, max_output_len=None):
    if max_output_len is None:
        max_output_len = args.max_output_len

    def output_filter(action):
        return len(action) <= max_output_len

    if args.dataset == 'SCAN':
        if args.use_scan_augmented:
            all_train_data = ScanAugmentedDataset(args.train_data_file, vocab, out_filter_fn=output_filter)
        else:
            all_train_data = ScanDataset(args.train_data_file, vocab, out_filter_fn=output_filter)
        if args.auto_val_split:
            split_id = int(0.8*len(all_train_data))
            train_data = [all_train_data[i] for i in range(split_id)]
            val_data = [all_train_data[i] for i in range(split_id,len(all_train_data))]
        else:
            train_data = all_train_data
            val_data = ScanDataset(args.val_data_file, vocab, out_filter_fn=output_filter)
        test_data = ScanDataset(args.test_data_file, vocab)

        print('Number of train examples:', len(train_data))
        print('Number of validation examples:', len(val_data))
        print('Number of test examples:', len(test_data))
    elif args.dataset == 'MT':
        train_data = MTDataset(args.train_data_file,vocab,args.flip)
        val_data = MTDataset(args.val_data_file,vocab,args.flip)
        test_data = MTDataset(args.test_data_file,vocab,args.flip)

    train_loader = DataLoader(train_data, args.batch_size,
                              shuffle=True, collate_fn=SCAN_collate)
    val_loader = DataLoader(val_data, args.batch_size,
                            shuffle=True, collate_fn=SCAN_collate)
    test_loader = DataLoader(test_data, args.batch_size,
                             shuffle=True, collate_fn=SCAN_collate)

    return train_loader, val_loader, test_loader

def main(args):
    if args.extra_config_path:
        experiment.log_asset(args.extra_config_path)
        yaml_config = yaml.load(open(args.extra_config_path))
        curriculum = yaml_config['curriculum']
    else:
        yaml_config = None
        curriculum = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Vocab
    with open(args.load_vocab_json,'r') as f:
        vocab = json.load(f)

    # Dataloaders
    train_loader, val_loader, test_loader = load_dataloaders(args, vocab)

    in_vocab_size = len(vocab['in_token_to_idx'])
    out_vocab_size = len(vocab['out_idx_to_token'])
    # Model
    if args.model == 'syntactic_attention':
        model = Seq2SeqSynAttn(in_vocab_size, args.m_hidden_dim, args.x_hidden_dim,
                            out_vocab_size, args.rnn_type, args.enc_n_layers, args.dec_n_layers,
                            args.dropout_p, args.seq_sem, args.syn_act,
                            args.sem_mlp, None, device)
    elif args.model == 'symbolic_operator':
        if args.fixed_gate:
            pretrained_gate = get_pretrained_gate(vocab)
        else:
            pretrained_gate = None
        model = SymbolicOperator(in_vocab_size, out_vocab_size,
            eos_idx=vocab['out_token_to_idx']['<EOS>'],
            max_program_steps=args.max_program_steps,
            pretrained_gate=pretrained_gate,
            gate_activation_train=args.gate_activation_train,
            gate_activation_eval=args.gate_activation_eval,
            gate_activation_temperature=args.gate_activation_temperature,
            read_activation_train=args.read_activation_train,
            read_activation_eval=args.read_activation_eval,
            write_activation_train=args.write_activation_train,
            write_activation_eval=args.write_activation_eval,
            scratch_max_len=50,
            use_adaptive_steps=args.use_adaptive_steps,
            keep_going_input=args.keep_going_input)
    else:
        raise ValueError('Invalid model name %s' % (args.model))

    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)

    # Loss function
    loss_fn = nn.NLLLoss(reduction='mean',ignore_index=-100)
    loss_fn = loss_fn.to(device)

    # Optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.learning_rate)

    # Training loop:
    iter = 0
    curriculum_idx = 0
    curriculum_iter = 1
    current_curriculum = None
    epoch_count = 0
    losses, base_losses, adaptive_steps_losses = [], [], []
    train_errors, val_errors, test_errors = [], [], []
    best_val_error = 1.1 # best validation error - for early stopping
    while iter < args.num_iters:
        epoch_count += 1

        if current_curriculum and 'max_output_len' in current_curriculum:
            train_loader, val_loader, test_loader = load_dataloaders(
                args, vocab, max_output_len=current_curriculum['max_output_len'])
        for sample_count,sample in enumerate(train_loader):
            if curriculum:
                current_curriculum = curriculum[curriculum_idx]
                if 'iterations' in current_curriculum and curriculum_iter >= current_curriculum['iterations']:
                    curriculum_idx += 1
                    curriculum_iter = 0
                curriculum_iter += 1
            else:
                current_curriculum = None
            iter += 1
            # Forward pass
            instructions, true_actions, _, _ = sample
            instructions = [ins.to(device) for ins in instructions]
            true_actions = [ta.to(device) for ta in true_actions]
            optimizer.zero_grad()
            if args.model == 'symbolic_operator':
                result = model(instructions, true_actions, extra_config=current_curriculum)
            else:
                result = model(instructions, true_actions)
            if args.use_adaptive_steps:
                actions, padded_true_actions, keep_going_loss = result
            else:
                actions, padded_true_actions = result
            # Compute NLLLoss
            true_actions = padded_true_actions.to(device)
            base_loss = loss_fn(actions,padded_true_actions)
            adaptive_steps_loss = torch.tensor([0], dtype=torch.float, device=device)
            if args.use_adaptive_steps:
                adaptive_steps_loss = args.adaptive_steps_loss_weight * keep_going_loss.mean()
            loss = base_loss + adaptive_steps_loss
            # Backward pass
            loss.backward()
            if args.clip_norm is not None:
                clip_grad_norm_(params,max_norm=args.clip_norm)
            optimizer.step()
            # Record loss
            if iter % args.record_loss_every == 0:
                loss_datapoint = loss.data.item()
                print('Epoch:', epoch_count,
                      'Iter:', iter,
                      'Loss:', loss_datapoint)
                losses.append(loss_datapoint)
                base_losses.append(base_loss.item())
                adaptive_steps_losses.append(adaptive_steps_loss.item())
        # Checkpoint
        last_epoch = (iter >= args.num_iters)
        if epoch_count % args.checkpoint_every == 0 or last_epoch:
            print("Checking training error...")
            train_error = check_accuracy(train_loader, model, device, args)
            print("Training error is ", train_error)
            train_errors.append(train_error)
            print("Checking validation error...")
            val_error = check_accuracy(val_loader, model, device, args)
            print("Validation error is ", val_error)
            val_errors.append(val_error)
            print("Checking test error...")
            test_error = check_accuracy(test_loader, model, device, args)
            print("Test error is ", test_error)
            test_errors.append(test_error)

            # Write stats file
            results_path = os.path.join(args.results_dir, experiment.get_key())
            if not os.path.isdir(results_path):
                os.mkdir(results_path)
            stats = {'loss_data':losses,
                     'train_errors':train_errors,
                     'val_errors':val_errors,
                     'test_errors':test_errors}
            results_file_name = '%s/%s' % (results_path,args.out_data_file)
            with open(results_file_name, 'w') as f:
                json.dump(stats, f)

            # Log metrics to comet
            metrics = {
                'epoch': epoch_count,
                'loss': np.mean(losses),
                'base_loss': np.mean(base_losses),
                'adaptive_steps_loss': np.mean(adaptive_steps_losses),
                'train_seq_acc': 1.0 - train_error,
                'val_seq_acc': 1.0 - val_error,
                'test_seq_acc': 1.0 - test_error
            }
            experiment.log_metrics(metrics)

            # Save model weights
            if args.save_all_checkpoints or val_error < best_val_error: # use val (not test) to decide to save
                best_val_error = val_error
                if args.checkpoint_dir is not None:
                    checkpoint_path = os.path.join(args.checkpoint_dir, experiment.get_key())
                    torch.save(model.state_dict(),
                               checkpoint_path)


def check_accuracy(dataloader, model, device, args):
    model.eval()
    with torch.no_grad():
        all_correct_trials = [] # list of booleans indicating whether correct
        for sample in dataloader:
            instructions, true_actions, _, _ = sample
            batch_size = len(instructions)
            out_vocab_size = model.out_vocab_size
            instructions = [ins.to(device) for ins in instructions]
            true_actions = [ta.to(device) for ta in true_actions]
            result = model(instructions, true_actions)
            if args.use_adaptive_steps:
                actions, padded_true_actions, keep_going_loss = result
            else:
                actions, padded_true_actions = result

            # Manually unpad with mask to compute accuracy
            mask = padded_true_actions == -100
            max_actions = torch.argmax(actions,dim=1)
            correct_actions = max_actions == padded_true_actions
            correct_actions = correct_actions + mask # Add boolean mask
            correct_actions = correct_actions.cpu().numpy()
            correct_trials = np.all(correct_actions,axis=1).tolist()
            all_correct_trials = all_correct_trials + correct_trials
    model.train()
    return 1.0 - np.mean(all_correct_trials)

if __name__ == '__main__':
    args = parser.parse_args()
    validate_args(parser, args)

    experiment.set_name(args.exp_name)
    log_comet_parameters(args)

    print(args)
    main(args)
