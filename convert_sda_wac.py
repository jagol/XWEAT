import re
import bz2


def convert(path_in, path_out):
    fin = bz2.BZ2File(path_in, 'r')
    fout = open(path_out, 'w')
    for line in fin:
        uline = line.decode('utf8')
        cline = re.sub(r'<.*>', '', uline).strip('\n')
        cline = re.sub(r'\t', '', cline)
        fout.write(cline + '\n')
    fin.close()
    fout.close()


if __name__ == '__main__':
    path_sde_wac = '/mnt/storage/clfiles/resources/data/corpora/DeWaC/unknown/sdewac-v3.corpus.bz2'
    pout = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/sde_wac_ospl.txt'
    convert(path_sde_wac, pout)
