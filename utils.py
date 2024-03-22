def write_log(s, path = 'log.out', prnt = True):
    ''
    
    f = open(path, "a")
    f.write('\n' + s)
    if prnt:
        print(s)
    f.close()

def load_node2vec(path: str):
    ''
    
    embeddings_dict = {}
    with open(path, "r") as f:
        for i, line in enumerate(tqdm(f)):
            vals = line.strip().split()
            embeddings_dict[vals[0]] = [float(x) for x in vals[1:]]

    print("### Node2Vec loaded")
    return embeddings_dict

def load_glove(path: str):
    ''
    
    embeddings_dict = {}
    with open(path, "r") as f:
        for i, line in enumerate(tqdm(f)):
            vals = line.rstrip().split()
            embeddings_dict[vals[0]] = [float(x) for x in vals[1:]]

    dict_keys = list(embeddings_dict.keys())
    print("### Glove loaded, vocab size {len(embeddings_dict.keys())}")
    return embeddings_dict, dict_keys

def load_entities(path: str):
    ''

    entities_dict = {}
    with open(path) as f:
        for line in f.readlines():
            fields = line.strip().split('\t')
            entities_dict[ fields[0] ] = fields[1]
    
    print('\nEntities loaded')
    return entities_dict

def load_relations(path: str):
    ''

    relations = []
    with open(path) as f:
        for line in f.readlines():
            relations.append(line.strip())
    
    print('\nRelations loaded')
    return relations