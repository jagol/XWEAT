from bert import load_model, load_tokenizer, encode, MODELS, TOKENIZERS


def load_sentences(fpath):
    with open(fpath) as f:
        for line in f:
            return {id_: sentence for id_, sentence in line.strip().split()}


def write_to_file(sentences, embeddings):
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Path to file containing sentences to embed.')
    parser.add_argument('-o', '--outptut', required=True, help='Path to output file where embeddings are written to.')
    parser.add_argument('-t', '--tokenizer', required=True, help='Name of Tokenizer to be used.')
    parser.add_argument('-m', '--model', required=True, help='Name of Model to be used.')
    args = parser.parse_args()
    sentences = load_sentences(args.input)
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.model)
    embeddings = encode(model, tokenizer, sentences)
    write_to_file(sentences, embeddings, args.outptut)


if __name__ == '__main__':
    main()
