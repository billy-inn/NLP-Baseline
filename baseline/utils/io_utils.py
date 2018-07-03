import json
import logging
import nltk

def read_snli(filename, lowercase):
    """
    Read a JSONL file with the SNLI corpus

    :param filename: path to the file
    :param lowercase: whether to convert text to lower case
    :return: a list of tuples (sent1, sent2, label)
    """

    logging.info('Reading SNLI data from %s' % filename)

    useful_data = []
    with open(filename, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            if lowercase:
                line = line.lower()
            data = json.loads(line)
            if data['gold_label'] == '-':
                continue

            sent1_parse = data['sentence1_parse']
            sent2_parse = data['sentence2_parse']
            label = data['gold_label']

            tree1 = nltk.Tree.fromstring(sent1_parse)
            tree2 = nltk.Tree.fromstring(sent2_parse)
            tokens1 = tree1.leaves()
            tokens2 = tree2.leaves()
            t = (tokens1, tokens2, label)
            useful_data.append(t)

    return useful_data
