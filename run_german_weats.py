import csv
from weat import run_weat_test

lang = 'de'
permutation_number = 100000
do_lower = True
similarity_type = 'cosine'
genders = ['both', 'male', 'female']
embedding_paths = {
    'wiki': '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/wiki_ospl_trimmed.txt.vec',
    'sde': '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/sde_wac_ospl_trimmed.txt.vec',
    'htb': '/mnt/storage/harlie/users/jgoldz/bias_germ_embeddings/data/hamburg_tb_ospl_trimmed.txt.vec',
}

test_ids = {
    # 'original_weats': ['weat_3', 'weat_4', 'weat_5'], too many words not in vocab
    'migrant_pleasant_unpleasant': [
        'weat_migrant_pleasant_unpleasant_1', 'weat_migrant_pleasant_unpleasant_2',
        'weat_migrant_pleasant_unpleasant_3', 'weat_migrant_pleasant_unpleasant_4',
        'weat_migrant_pleasant_unpleasant_5', 'weat_migrant_pleasant_unpleasant_6',
        'weat_migrant_pleasant_unpleasant_7', 'weat_migrant_pleasant_unpleasant_8',
        'weat_migrant_pleasant_unpleasant_9', 'weat_migrant_pleasant_unpleasant_10',
        'weat_migrant_pleasant_unpleasant_11', 'weat_migrant_pleasant_unpleasant_12',
        'weat_migrant_pleasant_unpleasant_13', 'weat_migrant_pleasant_unpleasant_14',
        'weat_migrant_pleasant_unpleasant_15', 'weat_migrant_pleasant_unpleasant_16',
        'weat_migrant_pleasant_unpleasant_17'
    ],
    'migrant_career_crime': [
        'weat_migrant_career_crime_1', 'weat_migrant_career_crime_2',
        'weat_migrant_career_crime_3', 'weat_migrant_career_crime_4',
        'weat_migrant_career_crime_5', 'weat_migrant_career_crime_6',
        'weat_migrant_career_crime_7', 'weat_migrant_career_crime_8',
        'weat_migrant_career_crime_9', 'weat_migrant_career_crime_10',
        'weat_migrant_career_crime_11', 'weat_migrant_career_crime_12',
        'weat_migrant_career_crime_13', 'weat_migrant_career_crime_14',
        'weat_migrant_career_crime_15', 'weat_migrant_career_crime_16',
        'weat_migrant_career_crime_17'
    ],
    'gender_career_family': [
        'weat_gender_career_family_1', 'weat_gender_career_family_2',
        'weat_gender_career_family_3', 'weat_gender_career_family_4',
        'weat_gender_career_family_5', 'weat_gender_career_family_6',
        'weat_gender_career_family_7'
    ],
    'anti-semitism': [
        'weat_jewish_pleasant_unpleasant_1', 'weat_jewish_career_crime_1',
        'weat_jewish_virtuous_greed_1'],
    'germanic_english': [
        'weat_germanic_english_pleasant_unpleasant', 'weat_germanic_english_career_crime'
    ]
}


def run_german_weats():
    fieldnames = ['test_id', 'embedding', 'gender', 'test-statistic', 'effect-size', 'p-value', 'warning']
    writer = csv.DictWriter(open('german_weat_results.csv', 'w'), fieldnames=fieldnames)
    writer.writeheader()
    for emb_type in embedding_paths:
        for cat in test_ids:
            if cat in ['migrant_pleasant_unpleasant', 'migrant_career_crime']:
                for gender in genders:
                    for test_id in test_ids[cat]:
                        print(10*'- ' + test_id + 10*' -')
                        print(f'CONFIG: {emb_type}, {gender}')
                        results = run_weat_test(test_id=test_id, embeddings=embedding_paths[emb_type],
                                                permutation_number=permutation_number, do_lower=do_lower,
                                                similarity_type=similarity_type, lang=lang, gender=gender)
                        writer.writerow({'test_id': test_id, 'embedding': emb_type, 'gender': gender,
                                         'test-statistic': results['test-statistic'],
                                         'effect-size': results['effect-size'], 'p-value': results['p-value'],
                                         'warning': results['warning']})
            else:
                for test_id in test_ids[cat]:
                    gender = 'both'
                    print(10 * '- ' + test_id + 10 * ' -')
                    results = run_weat_test(test_id=test_id, embeddings=embedding_paths[emb_type],
                                            permutation_number=permutation_number, do_lower=do_lower,
                                            similarity_type=similarity_type, lang=lang, gender=gender)
                    writer.writerow({'test_id': test_id, 'embedding': emb_type, 'gender': gender,
                                     'test-statistic': results['test-statistic'],
                                     'effect-size': results['effect-size'], 'p-value': results['p-value'],
                                     'warning': results['warning']})


if __name__ == '__main__':
    run_german_weats()
