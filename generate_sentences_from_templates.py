import os


def generate_cmds_names(data_dir, exec_dir):
    cmds = []
    os.chdir(data_dir)
    name_files = [fn for fn in os.listdir() if fn.endswith('_names.txt')]
    os.chdir(exec_dir)
    name_file_dict = {nf: [] for nf in name_files}
    for nf in name_file_dict:
        fn = nf[:-4]
        sent_fn = fn + '_sentences'
        sent_fnf = sent_fn + '_female'
        sent_fnm = sent_fn + '_male'
        name_file_dict[nf] = [fname + '.txt' for fname in [sent_fn, sent_fnf, sent_fnm]]
    for nf in name_file_dict:
        for sent_fn, ttype in zip(name_file_dict[nf], ['name', 'mname', 'fname']):
            cmds.append(f'python3 templates.py -i {data_dir}{nf} -o {data_dir}{sent_fn} -t {ttype}')
    return cmds


def main():
    data_dir = '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/word_lists/'
    exec_dir = '/home/user/jgoldz/bias/xweat'
    cmds = generate_cmds_names(data_dir, exec_dir)
    num_cmds = len(cmds)
    for i, cmd in enumerate(cmds):
        print(f'{i+1}/{num_cmds}: {cmd}')
        os.system(cmd)


if __name__ == '__main__':
    main()
