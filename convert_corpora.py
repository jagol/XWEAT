import os
import csv


def get_num_tokens(path_in):
    i = 0
    for _ in open(path_in):
        i += 1
    return i


def convert(path_in, path_out, num_tokens):
    with open(path_in, 'r', encoding='utf8') as fin, open(path_out, 'w', encoding='utf8') as fout:
        reader = csv.reader(fin, delimiter='\t')
        next(reader)
        cur_sent = []
        prev_sent_id = 1000000
        for i, row in enumerate(reader):
            if len(row) != 13:
                continue
            token = row[1]
            sent_id = row[-2]
            if sent_id != prev_sent_id:
                if len(cur_sent) > 4:
                    fout.write(' '.join(cur_sent) + '\n')
                cur_sent = [token]
                prev_sent_id = sent_id
            else:
                cur_sent.append(token)
            if i % 10000 == 0 and i != 0:
                print(f'Processed tokens: [{i}/{num_tokens}]\r', end='\r')
        print(f'Processed tokens: [{i}/{num_tokens}]')


def main():
    dir_path = '/home/janis/Dropbox/UZH/10._Semester/NLP_in_Context_of_AI/Paper/WEAT_Experiments/data'
    for fname_in in os.listdir(dir_path):
        if not fname_in.endswith('.tsv'):
            continue
        fname_out = fname_in.split('.')[0] + '_ospl.txt'
        path_in = os.path.join(dir_path, fname_in)
        path_out = os.path.join(dir_path, fname_out)
        print(f'Input file: {fname_in}')
        print(f'Output file: {fname_out}')
        print('Counting tokens...')
        num_tokens = get_num_tokens(path_in)
        print('Converting...')
        convert(path_in, path_out, num_tokens)
        print('Done.')


if __name__ == '__main__':
    main()
