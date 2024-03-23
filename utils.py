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
    """Writes a log record to harddrive.

    Writes a log record in a file in the harddrive, with options to specify
    the file name or whether to perform a print instruction.

    Args:
        s: A string of message to be recorded.
        path: A file path.
        prnt: A boolean indicating whether to print or not.

    Returns:
        None.

    Raises:
        IOError: An error occurred accessing the file.
    """
    
    f = open(path, "a")
    f.write('\n' + s)
    if prnt:
        print(s)
    f.close()
    

def load_node2vec(path: str):
    """Loads Node2Vec embeddings.

    Given a path to a Node2Vec generated file. The function loads the
    file of key-value store structure and returns it as a dictionary.

    Args:
        path: A string of a file path.

    Returns:
        Dictionary of nodes and their embeddings.

    Raises:
        FileNotFoundError: An error occurred accessing a file that does 
        not exist.
    """
    
    embeddings_dict = {}
    with open(path, "r") as f:
        for i, line in enumerate(tqdm(f)):
            vals = line.strip().split()
            embeddings_dict[vals[0]] = [float(x) for x in vals[1:]]

    print("### Node2Vec loaded")
    return embeddings_dict

def load_glove(path: str):
    """Loads Glove embeddings.

    Given a path to a Glove embeddings. The function loads the
    file of key-value store structure and returns it as a dictionary.

    Args:
        path: A string of a file path.

    Returns:
        Dictionary of words and their embeddings.
        Keys of the dictionary returned.

    Raises:
        FileNotFoundError: An error occurred accessing a file that does 
        not exist.
    """
    
    embeddings_dict = {}
    with open(path, "r") as f:
        for i, line in enumerate(tqdm(f)):
            vals = line.rstrip().split()
            embeddings_dict[vals[0]] = [float(x) for x in vals[1:]]

    dict_keys = list(embeddings_dict.keys())
    print("### Glove loaded, vocab size {len(embeddings_dict.keys())}")
    return embeddings_dict, dict_keys

def load_entities(path: str):
    """Loads entities dictionary.

    Given a path to an entities translation file. The function loads the
    file of entity IDs and their translation to names of text sequences.

    Args:
        path: A string of a file path.

    Returns:
        Dictionary of entities IDS as the keys and their corresponding
        names as the values.

    Raises:
        FileNotFoundError: An error occurred accessing a file that does 
        not exist.
    """

    entities_dict = {}
    with open(path) as f:
        for line in f.readlines():
            fields = line.strip().split('\t')
            entities_dict[ fields[0] ] = fields[1]
    
    print('\nEntities loaded')
    return entities_dict

def load_relations(path: str):
    """Loads relations dictionary.

    Given a path to relations translation file. The function loads the
    file of relation IDs and their translation of text sequences.

    Args:
        path: A string of a file path.

    Returns:
        Dictionary of relation IDS as the keys and their corresponding
        relation translations as the values.

    Raises:
        FileNotFoundError: An error occurred accessing a file that does 
        not exist.
    """

    relations = []
    with open(path) as f:
        for line in f.readlines():
            relations.append(line.strip())
    
    print('\nRelations loaded')
    return relations