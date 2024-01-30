from datasets import load_dataset
from itertools import product


def create_tsv_dataset(dataset_name='web_nlg', split_name='webnlg_challenge_2017'):
    dataset = load_dataset(dataset_name, split_name)

    pairs = {split: [] for split in dataset.keys()}
    max_len = 0
    for split in dataset.keys():
        for sample in dataset[split]:
            # The triple is in the form of: "Subject | Predicate | Object"
            # We will convert it to a tuple: (Subject, Predicate, Object)
            triples = sample['modified_triple_sets']['mtriple_set'][0]
            for i in range(len(triples)):
                triple = triples[i].split(' | ')
                triple = tuple(triple)
                triples[i] = triple

            texts = sample['lex']['text']

            pairs[split].append({
                'triples': triples,
                'texts': texts,
            })

            max_len = max(max_len, len(triple))

    # Store all in a list and create a mapping index between the triple and the text
    triples = {split: [] for split in pairs.keys()}
    texts = {split: [] for split in pairs.keys()}
    mapping = {split: {'triple': [], 'text': []} for split in pairs.keys()}
    for split in pairs.keys():
        for i, sample in enumerate(pairs[split]):
            for triple in sample['triples']:
                triples[split].append(triple)
                mapping[split]['triple'].append(i)
            for text in sample['texts']:
                texts[split].append(text)
                mapping[split]['text'].append(i)

    # Write to file
    # Write triples
    for split in triples.keys():
        with open('dataset/' + dataset_name + '_' + split_name + '_' + split + '_triples.tsv', 'w') as f:
            for triple in triples[split]:
                f.write('\t'.join(triple) + '\n')

    # Write texts
    for split in texts.keys():
        with open('dataset/' + dataset_name + '_' + split_name + '_' + split + '_texts.tsv', 'w') as f:
            for text in texts[split]:
                f.write(text + '\n')

    # TODO: Write mapping

    return triples, texts, mapping


create_tsv_dataset()
