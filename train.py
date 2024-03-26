"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import torch
import random
import numpy as np
from utils import Utils
from model import KGDataset, RPEST
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse

def seed_worker(worker_id):
    """Random seeding
    
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required = True, )
    parser.add_argument('--structural_path', required = True, )
    parser.add_argument('--textual_path', required = True, )
    parser.add_argument('--data_size', required = True, type = float)
    parser.add_argument('--learning_rate', required = True, type = float)
    parser.add_argument('--padding_size', required = True, type = int)
    parser.add_argument('--embedding_dimensions', required = True, type = int)
    parser.add_argument('--lstm_hidden_size', required = True, type = int)
    parser.add_argument('--lstm_layers', required = True, type = int)
    parser.add_argument('--dropout', required = True, type = float)
    parser.add_argument('--learning_decay', required = True, type = float)
    parser.add_argument('--batch_size', required = True, type = int)
    parser.add_argument('--epochs', required = True, type = int)
    parser.add_argument('--patience', required = True, type = int)
    parser.add_argument('--checkpoint_path', required = True, )
    parser.add_argument('--type_scalar', required = True, type = float)
    parser.add_argument('--verbose', required = True, type = bool)
    args = parser.parse_args()

    # Random seeds
    g = torch.Generator()
    g.manual_seed(0)

    # Load utilities
    utils = Utils()
    
    embeddings_dict, dict_keys = utils.load_glove(args.textual_path)
    relations = utils.load_relations(args.data_path + '/relations.txt')
    entities_dict = utils.load_entities(args.data_path + '/entity2text.txt')
    descriptions_dict = utils.load_entities(args.data_path + '/entity2textlong.txt')
    node2vec = utils.load_node2vec(args.structural_path)

    # Loading GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.write_log('\ndevice ' + str(device))

    # Load training set
    training_set = KGDataset(path = 'data/FB15K/train.tsv', entities_dict = entities_dict, descriptions_dict = descriptions_dict , relations = relations, device=device, args=args, node2vec = node2vec, embeddings_dict = embeddings_dict, dict_keys = dict_keys,)
    training_generator = DataLoader(training_set, batch_size = args.batch_size, worker_init_fn=seed_worker, generator=g,)

    validation_set = KGDataset(path = 'data/FB15K/dev.tsv', entities_dict = entities_dict, descriptions_dict = descriptions_dict , relations = relations, device=device, args=args, node2vec = node2vec, embeddings_dict = embeddings_dict, dict_keys = dict_keys,)
    validation_generator = DataLoader(validation_set, batch_size = args.batch_size, worker_init_fn=seed_worker, generator=g,)

    mymodel = RPEST(output_size = len(relations), args = args)
    loss_f = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr = args.learning_rate, )
    scheduler = lr_scheduler.StepLR(optimizer, gamma=args.learning_decay, step_size= 1)
    mymodel.to(device)

    # Print settings
    # for k, v in settings.items():
    #     utils.write_log(f'{k:<25} {v}')
    
    v_loss = 1_000_000
    no_change_counter = 1
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}\n-------------------------------')
        lr = optimizer.param_groups[0]['lr']
        mymodel.train()
        loop = tqdm(training_generator, disable = not args.verbose)
        losses = []

        # Loop over batches in an epoch using DataLoader
        for _, data in enumerate(loop):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            optimizer.zero_grad()
            predy = mymodel(data[0])
            loss = loss_f(predy, data[1])
            loss.backward()       
            optimizer.step()
            losses.append(loss.item())
        
        train_loss = sum(losses) / len(losses)
        lr = optimizer.param_groups[0]['lr']

        # Loop over batches in an epoch using DataLoader
        v_losses = []
        mymodel.eval()
        with torch.no_grad():

            for _, data in enumerate(validation_generator):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
                predy =  mymodel(data[0])
                loss = loss_f(predy, data[1])
                v_losses.append(loss)

        v_loss_epoch = sum(v_losses) / len(v_losses)
        utils.write_log(f'lr {lr:8f} train loss {train_loss:.8f} val loss {v_loss_epoch:.8f}')

        if v_loss - v_loss_epoch > 0.00001 :
            v_loss = v_loss_epoch
            no_change_counter = 0
            torch.save(mymodel.state_dict(), args.checkpoint_path)
        elif no_change_counter > args.patience - 1:
            break
        else:
            no_change_counter += 1
        
        scheduler.step()

if __name__ == "__main__": main()