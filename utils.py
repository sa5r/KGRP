"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

def write_log(s, path = 'log.out', prnt = True):
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the file.
    """
    
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