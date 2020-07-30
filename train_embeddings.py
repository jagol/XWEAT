import os
import argparse
import multiprocessing
from time import time
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class Sentences(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        for line in open(os.path.join(self.filepath)):
            yield line.split()


# def load_corpus_sentences(path_corpus):
#     with open(path_corpus, 'r', encoding='utf8') as f:
#         return [line.split(' ') for line in f]


class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1


def train_word2vec(path, fname_corpus):
    print(f'Train embeddings for corpus: {fname_corpus}')
    emb_dims = 300
    # sentences = Sentences(os.path.join(path, fname_corpus))
    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
    print('Start Training...')
    t = time()
    w2v_model = Word2Vec(min_count=5,
                         window=2,
                         size=emb_dims,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores-1,
                         iter=10,
                         corpus_file=os.path.join(path, fname_corpus),
                         compute_loss=True,
                         callbacks=[callback()]
                         )
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    # Build vocab
    # print('Building vocab...')
    # t = time()
    # w2v_model.build_vocab(Sentences(os.path.join(path, fname_corpus)), progress_per=10000)
    # print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    # train model
    # print('Training model...')
    # w2v_model.train(Sentences(os.path.join(path, fname_corpus)), total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    # print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    # save model as txt
    print('Saving vectors...')
    # vocab_size = len(w2v_model.wv.vocab)
    with open(os.path.join(path, fname_corpus.split('/')[-1]) + '.vec', 'w', encoding='utf8') as f:
        for i, word in enumerate(w2v_model.wv.vocab):
            f.write(word + ' ')
            for j, num in enumerate(w2v_model.wv[word]):
                f.write(str(num) + '\n') if j == emb_dims - 1 else f.write(str(num) + ' ')

# def train_w2c_in_all_corpora(path):
#     fns_corpora = [fn for fn in os.listdir(path) if fn.endswith('_ospl.txt')]
#     for fn in fns_corpora:
#         train_word2vec(path, fn)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', choices=['wiki', 'sde', 'htb'], required=True)
    parser.add_argument('--emb_dir', type=str, help='Directory where embeddings will be saved.', required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.emb_dir):
        raise Exception('Error. <emb_dir> is not a directory.')
    if args.corpus == 'wiki':
        path_corpus = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/wiki_corpus.txt'
    elif args.corpus == 'sde':
        path_corpus = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/sde_wac_ospl.txt'
    elif args.corpus == 'htb':
        path_corpus = ('/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/hamburg_dep_treebank/'
                       'hamburg_tb_ospl.txt')
    else:
        raise Exception("Error. arg <corpus> must be either 'wiki', 'sde' or 'htb'")
    train_word2vec(args.emb_dir, path_corpus)


if __name__ == '__main__':
    main()
