"""The Knowledge Graph completion model.

The Knowledge Graph completion model consists of three class. That are,
RPEST for the relation prediction component, KGDataset is the class that
defines the node representation component. SelfAttention is the class that
implements the self attention layer in the model's neural network. 

Typical usage example:

  model = RPEST(output_size = len(relations), args = args)
  model = model.train()
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from utils import Utils
import networkx as nx

class SelfAttention(torch.nn.Module):
    """The class that implements the self attention layer in the model's
    neural network.
    """
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = torch.nn.Linear(input_dim, input_dim)
        self.key = torch.nn.Linear(input_dim, input_dim)
        self.value = torch.nn.Linear(input_dim, input_dim)
        self.softmax = torch.nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted

class RPEST(torch.nn.Module):
    """The class for the relation prediction component in our model. The
    class implements a neural network of self attention layer, a flatten
    layer, and a dropout layer. Then Sigmoid is used as the activation
    function in the output layer.

    Typical usage example:

    model = RPEST(output_size = relations_len, args = args)
    model = model.train()
    """
    def __init__(self,output_size:int, args):
        ''
        
        super(RPEST, self).__init__()
        self.lstm = torch.nn.LSTM((args.embedding_dimensions + 1),
                                  args.lstm_hidden_size // 2,
                                  num_layers = args.lstm_layers,
                                  bidirectional=True, batch_first = True)
        self.attn = SelfAttention(args.lstm_hidden_size)
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(args.dropout)
        self.last = torch.nn.Linear(args.lstm_hidden_size * (args.padding_size + 2),
                                    output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.attn(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.last(x)
        x = self.sigmoid(x)
        return x

class KGDataset(Dataset):
    """The class that defines the node representation component.

    Typical usage example:
    training_set = KGDataset(path = 'data/FB15K/train.tsv',
                    entities_dict = entities_dict,
                    descriptions_dict = descriptions_dict ,
                    relations = relations, device=device,
                    args=args, node2vec = node2vec,
                    embeddings_dict = embeddings_dict,
                    dict_keys = dict_keys,)
    """

    def __init__(self, path: str,
        entities_dict: dict,
        descriptions_dict: dict,
        node2vec: list,
        relations: list,
        device,
        args,
        embeddings_dict: dict,
        dict_keys,
        ) -> None:
        """This constructor loads the necessary data.
        
        The cosntructor loads the structural and textual embedding
        dictionaries in addition to loading the graph.

        Args:

        Returns:
            None.
        
        Raises:

        """

        self.entities_dict = entities_dict
        self.descriptions_dict = descriptions_dict
        self.relations = relations
        self.device = device
        self.args = args
        self.node2vec = node2vec
        self.embeddings_dict = embeddings_dict
        self.dict_keys = dict_keys
        self.utils = Utils()

        # Read file
        f = open(path)
        file_lines = f.readlines()
        f.close()

        # Load lines based on the specified data amount
        self.lines = file_lines[: int(len(file_lines) * args.data_size) ]
        print(f'\nopened {path}' )
        self.G = nx.MultiGraph()
        for i in range( int( len(self.lines) ) ):
            items = self.lines[i].strip().split('\t')

            # Triple has 3 only
            assert len(items) == 3

            items = [p.strip() for p in items]

            head = items[0].strip()
            tail = items[2].strip()
            rel = items[1].strip()
            self.G.add_node(head)
            self.G.add_node(tail)
            self.G.add_edge(head, tail, name = rel)
        
        self.utils.write_log(str(self.G))
        
    def gettext(self, index):
        """Returns the complete details in a triple.
        
        Returns the relation text in a triple, in addition to
        the node IDs and their complete text content.

        Args:
            index: An integer of the triple index in the dataset.

        Returns:
            None.
        
        Raises:
        
        """

        fields = self.lines[index].strip().split('\t')
        head = self.entities_dict[fields[0]]
        tail = self.entities_dict[fields[2]]
        rel = fields[1]
        return head, rel, tail, fields[0],fields[2]
    
    def __len__(self) -> int:
        ''

        return len(self.lines)
    
    def get_glove_embedding(self, word):
        """Gets the embeddings of a word from the language model.
        
        Returns the embeddings as a float vector for a given word. The
        vector size is the number of dimensions in the language model. The
        function implements a greedy largest chuck search if the given
        word is not found in the language model vocabulary.

        Args:
            word: A string of a single word.

        Returns:
            A list of float values.
        
        Raises:
        
        """
        
        wrd = str(word).lower()
        if wrd in self.embeddings_dict:
            return self.embeddings_dict[wrd]
        
        # Greedy largest chunk
        lngth = len(wrd)
        for i in range(lngth - 1, 2, -1):
            for j in range(lngth - i + 1):
                sub_wrd = wrd[j:j + i]
                if sub_wrd in self.embeddings_dict:
                    return self.embeddings_dict[sub_wrd]
        
        char_embeddings = [
            self.embeddings_dict.setdefault(c, self.embeddings_dict[self.dict_keys[ord(c) % 100]]) for c in wrd
        ]
        averaged_embeddings = []
        for i in range(self.args.embedding_dimensions):
            tot = 0
            for j in range(len(char_embeddings)):
                tot += char_embeddings[j][i]
            averaged_embeddings.append(tot / self.args.embedding_dimensions)

        return averaged_embeddings
    
    def get_ent_embeddings(self, ent: str, is_head = False,
                           use_description = False):
        """Returns the node embeddings.
        
        Combines and returns the structural and textual embeddings for a
        node. Performs the padding for the returned matrix by appending
        vectors of zeros. Also performs the trimming of the extra words.

        Args:
            ent: The node ID as a string.
            is_head: A boolean to distinguish the head node from the
                    tail node. 
            use_description: A boolean to indicate the usage of node
                     instead of the node text.

        Returns:
            A matrix of combined textual and structural embeddings.
        
        Raises:
        
        """

        embeddings = []
        entity_text = self.entities_dict[ent]

        # Use description
        if use_description:
            entity_text = self.descriptions_dict[ent]

        words = entity_text.split()

        type_vector = [-1 * self.args.type_scalar]
        if is_head:
            type_vector = [self.args.type_scalar]

        # Check if longer than allowed
        max_size = self.args.padding_size // 2
        if len(words) > max_size:
            for j in range( max_size - 1 ):
                embeddings.append( self.get_glove_embedding(words[j]) + type_vector  )
            
            remaining_embeddings = []
            for j in range( max_size - 1, len(words) ):
                remaining_embeddings.append( self.get_glove_embedding(words[j]) )
            
            embeddings.append( np.mean(remaining_embeddings, axis=0).tolist() + type_vector )
        
        else:
            for j in range( len(words) ):
                embeddings.append( self.get_glove_embedding(words[j]) + type_vector  )

        if True: #N2V
            embeddings.append(self.node2vec[ent] + type_vector)
            
        # fill
        if len(embeddings) < self.args.padding_size // 2 + 1:
            filling = [((self.args.embedding_dimensions + 1) * [0.0])] * (self.args.padding_size // 2 + 1 - len(embeddings))
            embeddings.extend(filling)
        
                    
        # if is_head: # separator
        #     embeddings.append( self.args.DIMENSIONS * [-1] )        
            
        return embeddings
    
    def __getitem__(self, index):
        """Combines and returns the embeddings of two triple nodes.
        
        Combines and returns the embeddings of two nodes as the input
        for the model.

        Args:
            index: An integer of the input index in the dataset.

        Returns:
            A matrix of combined nodes pair embeddings.
        
        Raises:
        
        """

        fields = self.lines[index].strip().split('\t')
        rel = fields[1]

        assert len(fields) == 3

        # Prepare Y label
        relations_tagged = [0.0] * len(self.relations)
        rel_index = self.relations.index(rel)
        relations_tagged[ rel_index ] = 1.0

        embeddings = []
        head_embeddings = self.get_ent_embeddings(fields[0], is_head = True, )
        tail_embeddings = self.get_ent_embeddings(fields[2], is_head = False,  )
        embeddings.extend(head_embeddings)
        embeddings.extend(tail_embeddings)

        return torch.Tensor(embeddings),torch.Tensor(relations_tagged)