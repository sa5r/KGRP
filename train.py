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

class Train:
    """
    """

    def __init__(self, ):
        """Constructor
        """

        # Random seeds
        g = torch.Generator()
        g.manual_seed(0)

        # Load utilities
        self.utils = Utils()

        embeddings_dict, dict_keys = self.utils.load_glove(settings['glove_path'])
        relations = self.utils.load_relations(settings['relations_path'])
        entities_dict = self.utils.load_entities(settings['entities_path'])
        descriptions_dict = self.utils.load_entities(settings['descriptions_path'])
        node2vec = self.utils.load_node2vec(settings['n2v_path'])

        # Loading GPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        write_log('\ndevice ' + str(device))

        # Load training set
        training_set = KGDataset(path = 'data/FB15K/train.tsv', entities_dict = entities_dict, descriptions_dict = descriptions_dict , relations = relations, device=device, settings=settings, node2vec = node2vec, embeddings_dict = embeddings_dict, dict_keys = dict_keys,)
        training_generator = DataLoader(training_set, batch_size = settings['BATCH_SIZE'], worker_init_fn=seed_worker, generator=g,)

        validation_set = KGDataset(path = 'data/FB15K/dev.tsv', entities_dict = entities_dict, descriptions_dict = descriptions_dict , relations = relations, device=device, settings=settings, node2vec = node2vec, embeddings_dict = embeddings_dict, dict_keys = dict_keys,)
        validation_generator = DataLoader(validation_set, batch_size = settings['BATCH_SIZE'], worker_init_fn=seed_worker, generator=g,)

        mymodel = RPEST(output_size = len(relations), settings = settings)
        loss_f = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(mymodel.parameters(), lr = settings['lr'], )
        scheduler = lr_scheduler.StepLR(optimizer, gamma=settings['decay'], step_size= settings['stepping'])
        mymodel.to(device)

        # Print settings
        for k, v in settings.items():
            write_log(f'{k:<25} {v}')
        
        v_loss = 1_000_000
        no_change_counter = 1
        for epoch in range(settings['EPOCHS']):
            print(f'\nEpoch {epoch + 1}\n-------------------------------')
            lr = optimizer.param_groups[0]['lr']
            mymodel.train()
            loop = tqdm(training_generator, disable = not verbose)
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
            write_log(f'lr {lr:8f} train loss {train_loss:.8f} val loss {v_loss_epoch:.8f}')

            if v_loss - v_loss_epoch > 0.00001 :
                v_loss = v_loss_epoch
                no_change_counter = 0
                torch.save(mymodel.state_dict(), 'chkpnt.pt')
            elif no_change_counter > settings['PATIENCE'] - 1:
                break
            else:
                no_change_counter += 1
            
            scheduler.step()    
        
        mymodel = MyModel(output_size = len(relations), settings = settings)
        mymodel.to(device)
        mymodel.load_state_dict(torch.load('chkpnt.pt'))

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def main():
    """
    """

    pass

if __name__ == "__main__": main()