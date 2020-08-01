import os
import pickle
import argparse


def get_terms():
    """Fetch all terms used in weat."""
    terms = []
    dir_path_data = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/word_lists'
    for fn in os.listdir(dir_path_data):
        fpath = os.path.join(dir_path_data, fn)
        with open(fpath) as f:
            for line in f:
                terms.append(line.strip('\n'))
    dir_path_xweat_data = '/home/user/jgoldz/bias/xweat/data'
    for fn in os.listdir(dir_path_xweat_data):
        if fn.endswith('de.p'):
            fpath = os.path.join(dir_path_xweat_data, fn)
            with open(fpath, 'rb') as f:
                dict = pickle.load(f)
                for w1, w2 in dict.values():
                    terms.append(w1)
                    terms.append(w2)
    return [term for term in terms if term]


def load_filter_embeddings(fpath, terms):
    terms = [term.lower() for term in terms]
    embeddings = {}
    with open(fpath) as f:
        for line in f:
            fields = line.strip('\n').split(' ')
            word = fields[0]
            if word.lower() in terms:
                vec = [float(num) for num in fields[1:]]
                embeddings[word] = vec
    return embeddings


def write_filtered_embs_to_file(filtered_embs, fpath):
    with open(fpath, 'w') as f:
        for word in filtered_embs:
            f.write(word + ' ')
            for i, num in enumerate(filtered_embs[word]):
                if i == 299:
                    f.write(str(num) + '\n')
                else:
                    f.write(str(num) + ' ')


def trim_embeddings():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input file containing embeddings.')
    parser.add_argument('--output', required=True, help='Path to output file where embeddings are written to.')
    args = parser.parse_args()
    print('Loading terms...')
    terms = get_terms()
    print('Loading and filtering embeddings...')
    embeddings = load_filter_embeddings(args.input, terms)
    print('Writing embeddings to file...')
    write_filtered_embs_to_file(embeddings, args.output)


if __name__ == '__main__':
    trim_embeddings()
