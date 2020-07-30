def convert(path_in, path_out):
    with open(path_in, 'r') as fin, open(path_out, 'w') as fout:
        cur_sent = []
        for line in fin:
            line = line.strip('\n')
            if not line:
                continue
            if line.startswith('1\t') and cur_sent:
                out_line = ' '.join(cur_sent) + '\n'
                fout.write(out_line)
                cur_sent = []
            token = line.split('\t')[1]
            cur_sent.append(token)
    fin.close()
    fout.close()


if __name__ == '__main__':
    path_hamburg_tb = ('/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/hamburg_dep_treebank/'
                       'hamburg-dependency-treebank-conll/all_parts.conll')
    pout = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/hamburg_tb_ospl.txt'
    convert(path_hamburg_tb, pout)
