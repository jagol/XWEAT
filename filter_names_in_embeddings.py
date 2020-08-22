import argparse


path_sde = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/sde_wac_ospl.txt.vec'
path_htb = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/hamburg_tb_ospl.txt.vec'
path_wiki = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/wiki_ospl.txt.vec'


def filter_names(path_names_file, list_of_emb_paths):
    vocs = len(list_of_emb_paths) * [set()]
    for i, epath in enumerate(list_of_emb_paths):
        with open(epath) as f:
            lines = f.readlines()
            for line in lines:
                vocs[i].add(line.split(' ')[0])
    with open(path_names_file) as f:
        next(f)
        names = [name.strip('\n') for name in f]
    filtered_names = []
    for name in names:
        name_in_all = True
        for voc in vocs:
            if name not in voc:
                name_in_all = False
        if name_in_all:
            filtered_names.append(name)
    return filtered_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nf', '--names_file', required=True, help='Path to input file: list of names.')
    parser.add_argument('-o', '--out_file', required=True, help='Path to output file: filtered  list of names.')
    args = parser.parse_args()
    path_nf = args.names_file
    filtered_names = filter_names(path_nf, [path_sde, path_htb, path_wiki])
    with open(args.out_file, 'w') as f:
        for fn in filtered_names:
            f.write(fn + '\n')


if __name__ == '__main__':
    main()
