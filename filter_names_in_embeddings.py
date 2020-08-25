import argparse
from collections import defaultdict


path_sde = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/sde_wac_ospl.txt.count'
path_htb = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/hamburg_tb_ospl.txt.count'
path_wiki = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/wiki_ospl.txt.count'


def load_names(fpath):
    """Load names from file with 'source', 'male' and 'female'."""
    names_male = []
    names_female = []
    with open(fpath) as f:
        in_male = False
        in_female = False
        for line in f:
            line = line.strip('\n')
            if line.startswith('source') or not line:
                source_line = line
                continue
            if line.startswith('Male'):
                in_male = True
                continue
            elif line.startswith('Female'):
                in_female = True
                continue
            if in_male:
                names_male.append(line)
            elif in_female:
                names_female.append(line)
    return names_male, names_female, source_line


def load_vocs(list_of_voc_paths):
    """Load all vocabularies from count-files as dict {word:count}"""
    vocs = len(list_of_voc_paths) * [dict()]
    for i, vpath in enumerate(list_of_voc_paths):
        with open(vpath) as f:
            for line in f:
                word, count = line.strip('\n').split(' ')
                vocs[i][word] = count
    return vocs


def filter_names(names, vocs):
    """Filter out all names that are not in all vocs."""
    filtered_names = []
    for name in names:
        if all([name in voc for voc in vocs]):
            filtered_names.append(name)
    return filtered_names


def sort_by_count(names, vocs):
    """Sum up all voc-counts and sort names by summed up counts."""
    summed_voc = defaultdict()
    for voc in vocs:
        for word in voc:
            summed_voc[word] += voc[word]
    name_counts = sorted([(name, summed_voc[name]) for name in names], key=lambda x: x[1])
    return [item[0] for item in name_counts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nf', '--names_file', required=True, help='Path to input file: list of names.')
    parser.add_argument('-o', '--out_file', required=True, help='Path to output file: filtered  list of names.')
    parser.add_argument('-s', '--sort', required=True, type=bool, help='Sort names by occurrence count in corpora.')
    args = parser.parse_args()
    paths_vocs = [path_sde, path_htb, path_wiki]
    path_nf = args.names_file
    vocs = load_vocs(paths_vocs)
    names_male, names_female, source_line = load_names(path_nf)
    filtered_names_male = filter_names(names_male, vocs)
    filtered_names_female = filter_names(names_female, vocs)
    if args.sort:
        filtered_names_male = sort_by_count(filtered_names_male, vocs)
        filtered_names_female = sort_by_count(filtered_names_female, vocs)
    with open(args.out_file, 'w') as f:
        f.write(source_line)
        f.write('Male:\n')
        for fnm in filtered_names_male:
            f.write(fnm + '\n')
        f.write('Female:\n')
        for fnf in filtered_names_female:
            f.write(fnf + '\n')


if __name__ == '__main__':
    main()
