import argparse
from collections import defaultdict


paths = {
    'sde': ('/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/sde_wac_ospl.txt',
            '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/sde_wac_ospl.txt.count'),
    'htb': ('/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/hamburg_tb_ospl.txt',
            '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/hamburg_tb_ospl.txt.count'),
    'wiki': ('/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/wiki_ospl.txt',
             '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/wiki_ospl.txt.count')
}


def count_occurrences():
    for corpus_name, (fcorpus, fcount) in paths.items():
        print(f'Processing corpus {corpus_name}...')
        voc = defaultdict(int)  # {word: count}
        with open(fcorpus) as fin:
            for line in fin:
                line = line.lower().strip('\n')
                tokens = line.split(' ')
                for token in tokens:
                    voc[token] += 1
        print(f'Sorting keys for {corpus_name}...')
        sorted_keys = sorted(voc.keys())
        print(f'Writing counts for {corpus_name} to output file...')
        with open(fcount, 'w') as fout:
            for key in sorted_keys:
                fout.write(f'{key} {voc[key]}\n')


if __name__ == '__main__':
    count_occurrences()
