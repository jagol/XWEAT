import numpy as np
import random
from itertools import filterfalse
from itertools import combinations
import codecs
import utils
import os
import pickle
import logging
import argparse
import time
from collections import OrderedDict
import math
from time import time
from sklearn.metrics.pairwise import euclidean_distances


class XWEAT(object):
    """
    Perform WEAT (Word Embedding Association Test) bias tests on a language model.
    Follows from Caliskan et al 2017 (10.1126/science.aal4230).

    Credits: Basic implementation based on https://gist.github.com/SandyRogers/e5c2e938502a75dcae25216e4fae2da5
    """

    def __init__(self, gender, word_list_dir):
        self.gender = gender
        self.word_list_dir = word_list_dir
        self.wl_paths = self.get_paths(word_list_dir)
        self.embd_dict = None
        self.vocab = None
        self.embedding_matrix = None
        if self.gender == 'both':
            self.loading_func = self.load_names
        elif self.gender == 'female':
            self.loading_func = self.load_female_names
        elif self.gender == 'male':
            self.loading_func = self.loading_func
        else:
            raise Exception('Gender option not known.')

    def get_paths(self, word_list_dir):
        """Create paths for all word list files."""
        wl_paths = {}
        for fname in os.listdir(word_list_dir):
            key = fname.strip('.txt')
            wl_paths[key] = os.path.join(word_list_dir, fname)
        return wl_paths

    def set_embd_dict(self, embd_dict):
        self.embd_dict = embd_dict

    def _build_vocab_dict(self, vocab):
        self.vocab = OrderedDict()
        vocab = set(vocab)
        index = 0
        for term in vocab:
            if term in self.embd_dict:
                self.vocab[term] = index
                index += 1
            else:
                logging.warning('Not in vocab %s', term)

    def convert_by_vocab(self, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            if item in self.vocab:
                output.append(self.vocab[item])
            else:
                continue
        return output

    def _build_embedding_matrix(self):
        self.embedding_matrix = []
        for term, index in self.vocab.items():
            if term in self.embd_dict:
                self.embedding_matrix.append(self.embd_dict[term])
            else:
                raise AssertionError('This should not happen.')
        self.embd_dict = None

    @staticmethod
    def mat_normalize(mat, norm_order=2, axis=1):
        return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])

    def cosine(self, a, b):
        norm_a = self.mat_normalize(a)
        norm_b = self.mat_normalize(b)
        cos = np.dot(norm_a, np.transpose(norm_b))
        return cos

    def euclidean(self, a, b):
        norm_a = self.mat_normalize(a)
        norm_b = self.mat_normalize(b)
        distances = euclidean_distances(norm_a, norm_b)
        eucl = 1 / (1+distances)
        return eucl

    def csls(self, a, b, k=10):
        norm_a = self.mat_normalize(a)
        norm_b = self.mat_normalize(b)
        sims_local_a = np.dot(norm_a, np.transpose(norm_a))
        sims_local_b = np.dot(norm_b, np.transpose(norm_b))

        csls_norms_a = np.mean(np.sort(sims_local_a, axis=1)[:, -k - 1:-1], axis=1)
        csls_norms_b = np.mean(np.sort(sims_local_b, axis=1)[:, -k - 1:-1], axis=1)
        loc_sims = np.add(np.transpose(np.tile(csls_norms_a, (len(csls_norms_b), 1))),
                          np.tile(csls_norms_b, (len(csls_norms_a), 1)))

        return 2 * np.dot(norm_a, np.transpose(norm_b)) - loc_sims

    def _init_similarities(self, similarity_type):
        if similarity_type == 'cosine':
            self.similarities = self.cosine(self.embedding_matrix, self.embedding_matrix)
        elif similarity_type == 'csls':
            self.similarities = self.csls(self.embedding_matrix, self.embedding_matrix)
        elif similarity_type == 'euclidean':
            self.similarities = self.euclidean(self.embedding_matrix, self.embedding_matrix)
        else:
            raise NotImplementedError()

    @staticmethod
    def weat_1():
        """
        WEAT 1 (universally accepted): Targets 1=Flowers, Targets 2=Insects, Attributes 1=pleasant,
        Attributes 2=unpleasant
        :return: targets_1, targets_2, attributes_1, attributes_2
        """
        targets_1 = [
            'aster', 'clover', 'hyacinth', 'marigold', 'poppy', 'azalea', 'crocus', 'iris', 'orchid', 'rose',
            'blue-bell', 'daffodil', 'lilac', 'pansy', 'tulip', 'buttercup', 'daisy', 'lily', 'peony', 'violet',
            'carnation', 'gladiola', 'magnolia', 'petunia', 'zinnia'
        ]
        targets_2 = [
            'ant', 'caterpillar', 'flea', 'locust', 'spider', 'bedbug', 'centipede', 'fly', 'maggot', 'tarantula',
            'bee', 'cockroach', 'gnat', 'mosquito', 'termite', 'beetle', 'cricket', 'hornet', 'moth', 'wasp',
            'blackfly', 'dragonfly', 'horsefly', 'roach', 'weevil'
        ]
        attributes_1 = [
            'caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure',
            'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise',
            'family', 'happy', 'laughter', 'paradise', 'vacation'
        ]
        attributes_2 = [
            'abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink',
            'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty', 'ugly', 'cancer',
            'kill', 'rotten', 'vomit', 'agony', 'prison'
        ]
        return targets_1, targets_2, attributes_1, attributes_2

    # WEAT 2: Target 1=Musical instruments, Target 2= Weapons, Attributes 1: pleasant, Attributes 2=unpleasant
    @staticmethod
    def weat_2():
        targets_1 = [
            'bagpipe', 'cello', 'guitar', 'lute', 'trombone', 'banjo', 'clarinet', 'harmonica', 'mandolin',
            'trumpet', 'bassoon', 'drum', 'harp', 'oboe', 'tuba', 'bell', 'fiddle', 'harpsichord', 'piano',
            'viola', 'bongo', 'flute', 'horn', 'saxophone', 'violin']
        targets_2 = [
            'arrow', 'club', 'gun', 'missile', 'spear', 'axe', 'dagger', 'harpoon', 'pistol', 'sword', 'blade',
            'dynamite', 'hatchet', 'rifle', 'tank', 'bomb', 'firearm', 'knife', 'shotgun', 'teargas', 'cannon',
            'grenade', 'mace', 'slingshot', 'whip']
        attributes_1 = [
            'caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure',
            'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise',
            'family', 'happy', 'laughter', 'paradise', 'vacation'
        ]
        attributes_2 = [
            'abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink',
            'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty', 'ugly', 'cancer',
            'kill', 'rotten', 'vomit', 'agony', 'prison'
        ]
        return targets_1, targets_2, attributes_1, attributes_2

    # Here they deleted the infrequent african american names, and the same
    # number randomly choosen from the european american names
    def weat_3(self):
        # excluded in the original paper: Chip, Ian, Fred, Jed, Todd, Brandon, Wilbur, Sara, Amber,
        # Crystal, Meredith, Shannon, Donna,
        # Bobbie-Sue, Peggy, Sue-Ellen, Wendy
        # targets_1 = [
        #     'Adam', 'Harry', 'Josh', 'Roger', 'Alan', 'Frank', 'Justin', 'Ryan', 'Andrew', 'Jack', 'Matthew', 'Stephen',
        #     'Brad', 'Greg', 'Paul', 'Hank', 'Jonathan', 'Peter', 'Amanda', 'Courtney', 'Heather', 'Melanie', 'Katie',
        #     'Betsy', 'Kristin', 'Nancy', 'Stephanie', 'Ellen', 'Lauren',    'Colleen', 'Emily', 'Megan', 'Rachel',
        #     'Chip', 'Ian', 'Fred', 'Jed', 'Todd', 'Brandon', 'Wilbur', 'Sara', 'Amber', 'Crystal', 'Meredith',
        #     'Shannon', 'Donna', 'Bobbie-Sue', 'Peggy', 'Sue-Ellen', 'Wendy'
        # ]
        # excluded: Lerone, Percell, Rasaan, Rashaun, Everol, Terryl, Aiesha, Lashelle, Temeka, Tameisha, Teretha,
        # Latonya, Shanise,
        # Sharise, Tashika, Lashandra, Shavonn, Tawanda,
        targets_1 = self.load_names(self.wl_paths['german_names']) + \
                    self.load_names(self.wl_paths['swiss_names']) + \
                    self.load_names(self.wl_paths['austrian_names'])
        targets_2 = [
            'Alonzo', 'Jamel', 'Theo', 'Alphonse', 'Jerome', 'Leroy', 'Torrance', 'Darnell', 'Lamar', 'Lionel',
            'Tyree', 'Deion', 'Lamont', 'Malik', 'Terrence', 'Tyrone', 'Lavon', 'Marcellus', 'Wardell', 'Nichelle',
            'Shereen', 'Ebony', 'Latisha', 'Shaniqua', 'Jasmine', 'Tanisha', 'Tia', 'Lakisha', 'Latoya', 'Yolanda',
            'Malika', 'Yvette', 'Lerone', 'Percell', 'Rasaan', 'Rashaun', 'Everol', 'Terryl', 'Aiesha', 'Lashelle',
            'Temeka', 'Tameisha', 'Teretha', 'Latonya', 'Shanise', 'Sharise', 'Tashika', 'Lashandra', 'Shavonn',
            'Tawanda'
        ]
        attributes_1 = [
            'caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure',
            'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise',
            'family', 'happy', 'laughter', 'paradise', 'vacation'
        ]
        # same as before but agony and prison are replaced with bomb and evil
        attributes_2 = [
            'abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink',
            'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty', 'ugly', 'cancer',
            'kill', 'rotten', 'vomit', 'bomb', 'evil'
        ]
        return targets_1, targets_2, attributes_1, attributes_2

    # again: african american names vs. european american names and pleasant vs unpleasant but with different names
    def weat_4(self):
        # excluded as in the original paper: Jay, Kristen, (here only excluded in the glove experiments)
        # targets_1 = [
        #     'Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne',
        #     'Carrie', 'Emily', 'Jill', 'Laurie',    'Meredith', 'Sarah', 'Jay', 'Kristen'
        # ]
        targets_1 = self.load_names(self.wl_paths['german_names.txt']) + \
                    self.load_names(self.wl_paths['swiss_names.txt']) + \
                    self.load_names(self.wl_paths['austrian_names.txt'])
        # excluded in GloVe experiments:    Tremayne, Latonya,
        targets_2 = [
            'Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tyrone', 'Aisha', 'Ebony',
            'Keisha', 'Kenya',    'Lakisha', 'Latoya', 'Tamika', 'Tanisha', 'Tremayne', 'Latonya'
        ]
        attributes_1 = [
            'caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure',
            'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise',
            'family', 'happy', 'laughter', 'paradise', 'vacation'
        ]
        # same as before but agony and prison are replaced with bomb and evil
        attributes_2 = [
            'abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink',
            'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty', 'ugly', 'cancer',
            'kill', 'rotten', 'vomit', 'bomb', 'evil'
        ]
        return targets_1, targets_2, attributes_1, attributes_2

    # again african american vs european american names, but with different attributes
    def weat_5(self):
        # excluded as in the original paper: Jay, Kristen, (here only excluded in the glove experiments)
        # targets_1 = [
        #     'Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne',
        #     'Carrie', 'Emily', 'Jill', 'Laurie',    'Meredith', 'Sarah', 'Jay', 'Kristen'
        # ]
        targets_1 = self.load_names(self.wl_paths['german_names.txt']) + \
                    self.load_names(self.wl_paths['swiss_names.txt']) + \
                    self.load_names(self.wl_paths['austrian_names.txt'])
        # excluded in GloVe experiments:    Tremayne, Latonya,
        targets_2 = [
            'Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tyrone', 'Aisha', 'Ebony',
            'Keisha', 'Kenya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha', 'Tremayne', 'Latonya'
        ]
        attributes_1 = ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy']
        attributes_2 = ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']
        return targets_1, targets_2, attributes_1, attributes_2

    # sex- career
    @staticmethod
    def weat_6():
        targets_1 = ['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill']
        targets_2 = ['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna']
        attributes_1 = ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business',
                        'career']
        attributes_2 = ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']
        return targets_1, targets_2, attributes_1, attributes_2

    # sex - math
    @staticmethod
    def weat_7():
        targets_1 = ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition']
        targets_2 = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture']
        attributes_1 = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
        attributes_2 = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
        return targets_1, targets_2, attributes_1, attributes_2

    # sex - science
    @staticmethod
    def weat_8():
        targets_1 = ['science', 'technology', 'physics', 'chemistry', 'Einstein', 'NASA', 'experiment', 'astronomy']
        targets_2 = ['poetry', 'art', 'Shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama']
        attributes_1 = ['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him']
        attributes_2 = ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her']
        return targets_1, targets_2, attributes_1, attributes_2

    # desease - uncontrollability
    @staticmethod
    def weat_9():
        targets_1 = ['sad', 'hopeless', 'gloomy', 'tearful', 'miserable', 'depressed']
        targets_2 = ['sick', 'illness', 'influenza', 'disease', 'virus', 'cancer']
        # in w2v experiments, the've replaced short-term with short
        attributes_1 = ['impermanent', 'unstable', 'variable', 'fleeting', 'short-term', 'brief', 'occasional']
        attributes_2 = ['stable', 'always', 'constant', 'persistent', 'chronic', 'prolonged', 'forever']
        return targets_1, targets_2, attributes_1, attributes_2

    # old - pleasant
    @staticmethod
    def weat_10():
        targets_1 = ['Tiffany', 'Michelle', 'Cindy', 'Kristy', 'Brad', 'Eric', 'Joey', 'Billy']
        targets_2 = ['Ethel', 'Bernice', 'Gertrude', 'Agnes', 'Cecil', 'Wilbert', 'Mortimer', 'Edgar']
        attributes_1 = ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy']
        attributes_2 = ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']
        return targets_1, targets_2, attributes_1, attributes_2

    # missing from the original IAT: arab-muslim
    def load_names(self, fpath):
        title_lines = ['Male:', 'Female:']
        names = []
        with open(fpath) as f:
            for line in f:
                line = line.strip('\n')
                if line.startswith('source'):
                    continue
                if line in title_lines:
                    continue
                if not line:
                    continue
                names.append(line)
        return names

    def load_female_names(self, fpath):
        names = []
        in_fem_names = False
        with open(fpath) as f:
            for line in f:
                line = line.strip('\n')
                if line == 'Female:':
                    in_fem_names = True
                    continue
                if in_fem_names:
                    names.append(line)
            return names

    def load_male_names(self, fpath):
        names = []
        with open(fpath) as f:
            for line in f:
                line = line.strip('\n')
                if line.startswith('source:') or line == 'Male:':
                    continue
                names.append(line)
                if line == 'Female:':
                    return names

    def load_random_subset_of_names(self, fpath):
        raise NotImplementedError

    def load_random_subset_of_male_names(self, fpath):
        raise NotImplementedError

    def load_random_subset_of_female_names(self, fpath):
        raise NotImplementedError

    @staticmethod
    def load_word_list(fpath):
        with open(fpath) as f:
            return [line.strip('\n') for line in f if line.strip('\n')]

    # test anti-migrant bias
    # pleasant vs unpleasant
    # german names vs arabic names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_1(self):
        targets_1 = self.loading_func(self.wl_paths['german_names']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names'])
        targets_2 = self.loading_func(self.wl_paths['arabic_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs french names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_2(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['french_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs hebrew names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_3(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['hebrew_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs kosovo names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_4(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['kosovo_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs macedonian names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_5(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['macedonian_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs polish names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_6(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['polish_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs portuguese names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_7(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['portuguese_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs romanian names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_8(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['romanian_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs serbish names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_9(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['serbish_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs spanish names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_10(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['spanish_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs swiss names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_11(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['swiss_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # swiss names vs austrian names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_12(self):
        targets_1 = self.loading_func(self.wl_paths['swiss_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['austrian_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # austrian names vs german names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_13(self):
        targets_1 = self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['german_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs turkish names and pleasant vs unpleasant
    def weat_migrant_pleasant_unpleasant_14(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['turkish_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # west_european_pleasant_unpleasant
    def weat_migrant_pleasant_unpleasant_15(self):
        german = self.loading_func(self.wl_paths['german_names.txt'])
        swiss = self.loading_func(self.wl_paths['swiss_names.txt'])
        austrian = self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_1 = german + swiss + austrian
        french = self.loading_func(self.wl_paths['french_names.txt'])
        italian = self.loading_func(self.wl_paths['italian_names.txt'])
        portuguese = self.loading_func(self.wl_paths['portuguese_names.txt'])
        spanish = self.loading_func(self.wl_paths['spanish_names.txt'])
        targets_2 = french + italian + portuguese + spanish
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_pleasant_unpleasant_16(self):
        # weat_east_european_pleasant_unpleasant
        german = self.loading_func(self.wl_paths['german_names.txt'])
        swiss = self.loading_func(self.wl_paths['swiss_names.txt'])
        austrian = self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_1 = german + swiss + austrian
        serbish = self.loading_func(self.wl_paths['serbish_names.txt'])
        romanian = self.loading_func(self.wl_paths['romanian_names.txt'])
        polish = self.loading_func(self.wl_paths['polish_names.txt'])
        macedonian = self.loading_func(self.wl_paths['macedonian_names.txt'])
        kosovo = self.loading_func(self.wl_paths['kosovo_names.txt'])
        bosnian = self.loading_func(self.wl_paths['bosnian_names.txt'])
        croatian = self.loading_func(self.wl_paths['croatian_names.txt'])
        hungarian = self.loading_func(self.wl_paths['hungarian_names.txt'])
        slovak = self.loading_func(self.wl_paths['slovak_names.txt'])
        targets_2 = serbish + romanian + polish + macedonian + kosovo + bosnian + croatian + hungarian + slovak
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_pleasant_unpleasant_17(self):
        # weat_middle_eastern_pleasant_unpleasant
        german = self.loading_func(self.wl_paths['german_names.txt'])
        swiss = self.loading_func(self.wl_paths['swiss_names.txt'])
        austrian = self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_1 = german + swiss + austrian
        afghani = self.loading_func(self.wl_paths['afghani_names.txt'])
        syrian = self.loading_func(self.wl_paths['syrian_names.txt'])
        turkish = self.loading_func(self.wl_paths['turkish_names.txt'])
        # arabic = self.loading_func(self.wl_paths['arabic_names.txt'])
        targets_2 = afghani + syrian + turkish
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # career vs crime
    # german names vs union of various eastern names and career vs crime
    def weat_migrant_career_crime_1(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['turkish_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_2(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['serbish_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_3(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['romanian_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_4(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['polish_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_5(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['macedonian_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_6(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['kosovo_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_7(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['arabic_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_8(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['french_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_9(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['spanish_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_10(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['portuguese_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_11(self):
        # weat_east_european_career_crime
        german = self.loading_func(self.wl_paths['german_names.txt'])
        swiss = self.loading_func(self.wl_paths['swiss_names.txt'])
        austrian = self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_1 = german + swiss + austrian

        serbish = self.loading_func(self.wl_paths['serbish_names.txt'])
        romanian = self.loading_func(self.wl_paths['romanian_names.txt'])
        polish = self.loading_func(self.wl_paths['polish_names.txt'])
        macedonian = self.loading_func(self.wl_paths['macedonian_names.txt'])
        kosovo = self.loading_func(self.wl_paths['kosovo_names.txt'])
        bosnian = self.loading_func(self.wl_paths['bosnian_names.txt'])
        croatian = self.loading_func(self.wl_paths['croatian_names.txt'])
        hungarian = self.loading_func(self.wl_paths['hungarian_names.txt'])
        slovak = self.loading_func(self.wl_paths['slovak_names.txt'])
        targets_2 = serbish + romanian + polish + macedonian + kosovo + bosnian + croatian + hungarian + slovak

        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_12(self):
        # weat_middle_eastern_career_crime
        german = self.loading_func(self.wl_paths['german_names.txt'])
        swiss = self.loading_func(self.wl_paths['swiss_names.txt'])
        austrian = self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_1 = german + swiss + austrian

        afghani = self.loading_func(self.wl_paths['afghani_names.txt'])
        syrian = self.loading_func(self.wl_paths['syrian_names.txt'])
        turkish = self.loading_func(self.wl_paths['turkish_names.txt'])
        # arabic = self.loading_func(self.wl_paths['arabic_names.txt'])
        targets_2 = afghani + syrian + turkish

        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_migrant_career_crime_13(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt']) + \
                    self.loading_func(self.wl_paths['swiss_names.txt']) + \
                    self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['hebrew_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # german names vs swiss names and career vs crime
    def weat_migrant_career_crime_14(self):
        targets_1 = self.loading_func(self.wl_paths['german_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['swiss_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # swiss names vs austrian names and career vs crime
    def weat_migrant_career_crime_15(self):
        targets_1 = self.loading_func(self.wl_paths['swiss_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['austrian_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # austrian names vs german names and career vs crime
    def weat_migrant_career_crime_16(self):
        targets_1 = self.loading_func(self.wl_paths['austrian_names.txt'])
        targets_2 = self.loading_func(self.wl_paths['german_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime_german.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # --- gender tests ---
    # germanic - career vs family
    def weat_gender_career_family_1(self):
        # weat_male_german_female_german_career_family
        targets_1 = self.load_male_names(self.wl_paths['german_names.txt'])
        targets_2 = self.load_female_names(self.wl_paths['german_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['family.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_gender_career_family_2(self):
        # weat_male_swiss_female_swiss_career_family
        targets_1 = self.load_male_names(self.wl_paths['swiss_names.txt'])
        targets_2 = self.load_female_names(self.wl_paths['swiss_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['family.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_gender_career_family_3(self):
        # weat_male_austrian_female_austrian_career_family
        targets_1 = self.load_male_names(self.wl_paths['austrian_names.txt'])
        targets_2 = self.load_female_names(self.wl_paths['austrian_names.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['family.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_gender_career_family_4(self):
        # weat_male_germanic_female_germanic_career_family
        mgerman = self.load_male_names(self.wl_paths['german_names.txt'])
        mswiss = self.load_male_names(self.wl_paths['swiss_names.txt'])
        maustrian = self.load_male_names(self.wl_paths['austrian_names.txt'])
        targets_1 = mgerman + mswiss + maustrian
        fgerman = self.loading_func(self.wl_paths['german_names.txt'])
        fswiss = self.load_female_names(self.wl_paths['swiss_names.txt'])
        faustrian = self.load_female_names(self.wl_paths['austrian_names.txt'])
        targets_2 = fgerman + fswiss + faustrian
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['family.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # gender tests for eastern european names
    def weat_gender_career_family_5(self):
        # weat_male_eastern_eu_female_eastern_eu_career_family
        mserbish = self.load_male_names(self.wl_paths['serbish_names.txt'])
        mromanian = self.load_male_names(self.wl_paths['romanian_names.txt'])
        mpolish = self.load_male_names(self.wl_paths['polish_names.txt'])
        mmacedonian = self.load_male_names(self.wl_paths['macedonian_names.txt'])
        mkosovo = self.load_male_names(self.wl_paths['kosovo_names.txt'])
        mbosnian = self.load_male_names(self.wl_paths['bosnian_names.txt'])
        mcroatian = self.load_male_names(self.wl_paths['croatian_names.txt'])
        mhungarian = self.load_male_names(self.wl_paths['hungarian_names.txt'])
        mslovak = self.load_male_names(self.wl_paths['slovak_names.txt'])
        targets_1 = mserbish + mromanian + mpolish + mmacedonian + mkosovo + mbosnian + mcroatian + mcroatian + \
                    mhungarian + mslovak
        fserbish = self.load_female_names(self.wl_paths['serbish_names.txt'])
        fromanian = self.load_female_names(self.wl_paths['romanian_names.txt'])
        fpolish = self.load_female_names(self.wl_paths['polish_names.txt'])
        fmacedonian = self.load_female_names(self.wl_paths['macedonian_names.txt'])
        fkosovo = self.load_female_names(self.wl_paths['kosovo_names.txt'])
        fbosnian = self.load_female_names(self.wl_paths['bosnian_names.txt'])
        fcroatian = self.load_female_names(self.wl_paths['croatian_names.txt'])
        fhungarian = self.load_female_names(self.wl_paths['hungarian_names.txt'])
        fslovak = self.load_female_names(self.wl_paths['slovak_names.txt'])
        targets_2 = fserbish + fromanian + fpolish + fmacedonian + fkosovo + fbosnian + fcroatian + fhungarian + fslovak
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['family.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # gender tests for western european names
    def weat_gender_career_family_6(self):
        # weat_male_west_european_female_western_european_career_family
        mfrench = self.load_male_names(self.wl_paths['french_names.txt'])
        mitalian = self.load_male_names(self.wl_paths['italian_names.txt'])
        mportuguese = self.load_male_names(self.wl_paths['portuguese_names.txt'])
        mspanish = self.load_male_names(self.wl_paths['spanish_names.txt'])
        targets_1 = mfrench + mitalian + mportuguese + mspanish
        ffrench = self.load_male_names(self.wl_paths['french_names.txt'])
        fitalian = self.load_male_names(self.wl_paths['italian_names.txt'])
        fportuguese = self.load_male_names(self.wl_paths['portuguese_names.txt'])
        fspanish = self.load_male_names(self.wl_paths['spanish_names.txt'])
        targets_2 = ffrench + fitalian + fportuguese + fspanish
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['family.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # gender tests for middle eastern names
    def weat_gender_career_family_7(self):
        # weat_male_middle_eastern_female_middle_eastern_career_family
        mafghani = self.load_male_names(self.wl_paths['afghani_names.txt'])
        msyrian = self.load_male_names(self.wl_paths['syrian_names.txt'])
        mturkish = self.load_male_names(self.wl_paths['turkish_names.txt'])
        targets_1 = mafghani + msyrian + mturkish
        fafghani = self.load_female_names(self.wl_paths['afghani_names.txt'])
        fsyrian = self.load_female_names(self.wl_paths['syrian_names.txt'])
        fturkish = self.load_female_names(self.wl_paths['turkish_names.txt'])
        targets_2 = fafghani + fsyrian + fturkish
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['family.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # tests for anti-semitism
    def weat_jewish_pleasant_unpleasant_1(self):
        # weat_german_jewish_pleasant_unpleasant
        targets_1 = self.load_names(self.wl_paths['german_lastnames.txt'])
        targets_2 = self.load_names(self.wl_paths['jewish_lastnames_in_all_voc.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['pleasant.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['unpleasant.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    def weat_jewish_career_crime_1(self):
        # weat_german_jewish_career_crime
        targets_1 = self.load_names(self.wl_paths['german_lastnames.txt'])
        targets_2 = self.load_names(self.wl_paths['jewish_lastnames_in_all_voc.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['career.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['crime.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # greedy vs virtuous TODO: load name subset
    def weat_jewish_virtuous_greed_1(self):
        # weat_german_jewish_greed_stereotype
        targets_1 = self.load_names(self.wl_paths['german_lastnames.txt'])
        targets_2 = self.load_names(self.wl_paths['jewish_lastnames_in_all_voc.txt'])
        attributes_1 = self.load_word_list(self.wl_paths['virtuous.txt'])
        attributes_2 = self.load_word_list(self.wl_paths['greed.txt'])
        return targets_1, targets_2, attributes_1, attributes_2

    # occupations
    def wefat_1(self):
        # occupations derived from th bureau of labor statistics
        targets = [
            'technician', 'accountant', 'supervisor', 'engineer', 'worker', 'educator', 'clerk', 'counselor',
            'inspector', 'mechanic', 'manager', 'therapist', 'administrator', 'salesperson', 'receptionist',
            'librarian', 'advisor', 'pharmacist', 'janitor', 'psychologist', 'physician', 'carpenter', 'nurse',
            'investigator', 'bartender', 'specialist', 'electrician', 'officer', 'pathologist', 'teacher', 'lawyer',
            'planner', 'practitioner', 'plumber', 'instructor', 'surgeon', 'veterinarian', 'paramedic', 'examiner',
            'chemist', 'machinist', 'appraiser', 'nutritionist', 'architect', 'hairdresser', 'baker', 'programmer',
            'paralegal', 'hygienist', 'scientist'
        ]
        attributes_1 = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
        attributes_2 = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
        return targets, attributes_1, attributes_2

    # androgynous names
    def wefat_2(self):
        targets = [
            'Kelly', 'Tracy', 'Jamie', 'Jackie', 'Jesse', 'Courtney', 'Lynn', 'Taylor', 'Leslie', 'Shannon',
            'Stacey', 'Jessie', 'Shawn', 'Stacy', 'Casey', 'Bobby', 'Terry', 'Lee', 'Ashley', 'Eddie', 'Chris', 'Jody',
            'Pat', 'Carey', 'Willie', 'Morgan', 'Robbie', 'Joan', 'Alexis', 'Kris', 'Frankie', 'Bobbie', 'Dale',
            'Robin', 'Billie', 'Adrian', 'Kim', 'Jaime', 'Jean', 'Francis', 'Marion', 'Dana', 'Rene', 'Johnnie',
            'Jordan', 'Carmen', 'Ollie', 'Dominique', 'Jimmie', 'Shelby'
        ]
        attributes_1 = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
        attributes_2 = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
        return targets, attributes_1, attributes_2

    def similarity_precomputed_sims(self, w1, w2, type='cosine'):
        return self.similarities[w1, w2]

    def word_association_with_attribute_precomputed_sims(self, w, A, B):
        return np.mean([self.similarity_precomputed_sims(w, a) for a in A]) - np.mean([self.similarity_precomputed_sims(w, b) for b in B])


    def differential_association_precomputed_sims(self, T1, T2, A1, A2):
        return np.sum([self.word_association_with_attribute_precomputed_sims(t1, A1, A2) for t1 in T1]) \
                     - np.sum([self.word_association_with_attribute_precomputed_sims(t2, A1, A2) for t2 in T2])

    def weat_effect_size_precomputed_sims(self, T1, T2, A1, A2):
        return (
                         np.mean([self.word_association_with_attribute_precomputed_sims(t1, A1, A2) for t1 in T1]) -
                         np.mean([self.word_association_with_attribute_precomputed_sims(t2, A1, A2) for t2 in T2])
                     ) / np.std([self.word_association_with_attribute_precomputed_sims(w, A1, A2) for w in T1 + T2])

    def _random_permutation(self, iterable, r=None):
        pool = tuple(iterable)
        r = len(pool) if r is None else r
        return tuple(random.sample(pool, r))

    def weat_p_value_precomputed_sims(self, T1, T2, A1, A2, sample):
        logging.info('Calculating p value ... ')
        size_of_permutation = min(len(T1), len(T2))
        T1_T2 = T1 + T2
        observed_test_stats_over_permutations = []
        total_possible_permutations = math.factorial(len(T1_T2)) / math.factorial(size_of_permutation) / \
                                      math.factorial((len(T1_T2)-size_of_permutation))
        logging.info('Number of possible permutations: %d', total_possible_permutations)
        if not sample or sample >= total_possible_permutations:
            permutations = combinations(T1_T2, size_of_permutation)
        else:
            logging.info('Computing randomly first %d permutations', sample)
            permutations = set()
            while len(permutations) < sample:
                permutations.add(tuple(sorted(self._random_permutation(T1_T2, size_of_permutation))))

        for Xi in permutations:
            Yi = filterfalse(lambda w: w in Xi, T1_T2)
            observed_test_stats_over_permutations.append(self.differential_association_precomputed_sims(Xi, Yi, A1, A2))
            if len(observed_test_stats_over_permutations) % 100000 == 0:
                logging.info('Iteration %s finished', str(len(observed_test_stats_over_permutations)))
        unperturbed = self.differential_association_precomputed_sims(T1, T2, A1, A2)
        is_over = np.array([o > unperturbed for o in observed_test_stats_over_permutations])
        return is_over.sum() / is_over.size

    def weat_stats_precomputed_sims(self, T1, T2, A1, A2, sample_p=None):
        test_statistic = self.differential_association_precomputed_sims(T1, T2, A1, A2)
        effect_size = self.weat_effect_size_precomputed_sims(T1, T2, A1, A2)
        p = self.weat_p_value_precomputed_sims(T1, T2, A1, A2, sample=sample_p)
        return test_statistic, effect_size, p

    def _create_vocab(self):
        """
        >>> weat = XWEAT(None); weat._create_vocab()
        :return: all
        """
        all_sets = []
        for i in range(1, 10):
            t1, t2, a1, a2 = getattr(self, 'weat_' + str(i))()
            all_sets = all_sets + t1 + t2 + a1 + a2
        for i in range(1, 2):
            t1, a1, a2 = getattr(self, 'wefat_' + str(i))()
            all_sets = all_sets + t1 + a1 + a2
        all_sets = set(all_sets)
        return all_sets

    def _output_vocab(self, path='./data/vocab_en.txt'):
        """
        >>> weat = XWEAT(None); weat._output_vocab()
        """
        vocab = self._create_vocab()
        with codecs.open(path, 'w', 'utf8') as f:
            for w in vocab:
                f.write(w)
                f.write('\n')
            f.close()

    def run_test_precomputed_sims(self, target_1, target_2, attributes_1, attributes_2, sample_p=None,
                                  similarity_type='cosine'):
        """Run the WEAT test for differential association between two
        sets of target words and two sets of attributes.

        RETURNS:
                (d, e, p). A tuple of floats, where d is the WEAT Test statistic,
                e is the effect size, and p is the one-sided p-value measuring the
                (un)likeliness of the null hypothesis (which is that there is no
                difference in association between the two target word sets and
                the attributes).

                If e is large and p small, then differences in the model between
                the attribute word sets match differences between the targets.
        """
        vocab = target_1 + target_2 + attributes_1 + attributes_2
        self._build_vocab_dict(vocab)
        T1 = self.convert_by_vocab(target_1)
        T2 = self.convert_by_vocab(target_2)
        A1 = self.convert_by_vocab(attributes_1)
        A2 = self.convert_by_vocab(attributes_2)
        while len(T1) < len(T2):
            logging.info('Popped T2 %d', T2[-1])
            T2.pop(-1)
        while len(T2) < len(T1):
            logging.info('Popped T1 %d', T1[-1])
            T1.pop(-1)
        while len(A1) < len(A2):
            logging.info('Popped A2 %d', A2[-1])
            A2.pop(-1)
        while len(A2) < len(A1):
            logging.info('Popped A1 %d', A1[-1])
            A1.pop(-1)
        assert len(T1) == len(T2)
        assert len(A1) == len(A2)
        self._build_embedding_matrix()
        self._init_similarities(similarity_type)
        return self.weat_stats_precomputed_sims(T1, T2, A1, A2, sample_p)

    def _parse_translations(self, path='./data/vocab_en_de.csv', new_path='./data/vocab_dict_en_de.p',
                            is_russian=False):
        """
        :param path: path of the csv file edited by our translators
        :param new_path: path of the clean dict to save
        >>> XWEAT()._parse_translations(is_russian=False)
        293
        """
        # This code probably does not work for the russian code, as dmitry did use other columns for his corrections
        with codecs.open(path, 'r', 'utf8') as f:
            translation_dict = {}
            for line in f.readlines():
                parts = line.split(',')
                en = parts[0]
                if en == '' or en[0].isupper():
                    continue
                else:
                    if is_russian and parts[3] != '\n' and parts[3] != '\r\n' and parts[3] != '\r':
                            other_m = parts[2]
                            other_f = parts[3].strip()
                            translation_dict[en] = (other_m, other_f)
                    else:
                        other_m = parts[1].strip()
                        other_f = None
                        if len(parts) > 2 and parts[2] != '\n' and parts[2] != '\r\n' and parts[2] != '\r' and \
                                parts[2] != '':
                            other_f = parts[2].strip()
                        translation_dict[en] = (other_m, other_f)
            pickle.dump(translation_dict, open(new_path, 'wb'))
            return len(translation_dict)

def load_vocab_goran(path):
    return pickle.load(open(path, 'rb'))

def load_vectors_goran(path):
    return np.load(path)

def load_embedding_dict(vocab_path='', vector_path='', embeddings_path='', glove=False, postspec=False):
    """
    >>> _load_embedding_dict()
    :param vocab_path:
    :param vector_path:
    :return: embd_dict
    """
    if glove and postspec:
        raise ValueError('Glove and postspec cannot both be true')
    elif glove:
        if os.name == 'nt':
            embd_dict = utils.load_embeddings('C:/Users/anlausch/workspace/embedding_files/glove.6B/glove.6B.300d.txt',
                                                                                word2vec=False)
        else:
            embd_dict = utils.load_embeddings('/work/anlausch/glove.6B.300d.txt', word2vec=False)
        return embd_dict
    elif postspec:
        embd_dict_temp = utils.load_embeddings('/work/anlausch/ft_postspec.txt', word2vec=False)
        embd_dict = {}
        for key, value in embd_dict_temp.items():
            embd_dict[key.split('en_')[1]] = value
        assert('test' in embd_dict)
        assert ('house' in embd_dict)
        return embd_dict
    elif embeddings_path != '':
        embd_dict = utils.load_embeddings(embeddings_path, word2vec=False)
        return embd_dict
    else:
        embd_dict = {}
        vocab = load_vocab_goran(vocab_path)
        vectors = load_vectors_goran(vector_path)
        for term, index in vocab.items():
            embd_dict[term] = vectors[index]
        assert len(embd_dict) == len(vocab)
        return embd_dict

def translate(translation_dict, terms):
    translation = []
    for t in terms:
        if t in translation_dict or t.lower() in translation_dict:
            if t.lower() in translation_dict:
                male, female = translation_dict[t.lower()]
            elif t in translation_dict:
                male, female = translation_dict[t]
            if female is None or female is '':
                translation.append(male)
            else:
                translation.append(male)
                translation.append(female)
        else:
            translation.append(t)
    translation = list(set(translation))
    return translation

def compute_oov_percentage():
    """
    >>> compute_oov_percentage()
    :return:
    """
    with codecs.open('./results/oov_short.txt', 'w', 'utf8') as f:
        for test in range(1,11):
            f.write('Test %d \n' % test)
            targets_1, targets_2, attributes_1, attributes_2 = XWEAT().__getattribute__('weat_' + str(test))()
            vocab = targets_1 + targets_2 + attributes_1 + attributes_2
            vocab = [t.lower() for t in vocab]
            # f.write('English vocab: %s \n' % str(vocab))
            for language in ['en', 'es', 'de', 'tr', 'ru', 'hr', 'it']:
                if language != 'en':
                    # f.write('Translating terms from en to %s\n' % language)
                    translation_dict = load_vocab_goran('./data/vocab_dict_en_' + language + '.p')
                    vocab_translated = translate(translation_dict, vocab)
                    vocab_translated = [t.lower() for t in vocab_translated]
                    # f.write('Translated terms %s\n' % str(vocab))
                embd_dict = load_embedding_dict(
                    vocab_path='/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/'
                               'ft.wiki.'+language+'.300.vocab',
                    vector_path='/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/'
                                'ft.wiki.'+language+'.300.vectors'
                )
                ins=[]
                not_ins=[]
                if language != 'en':
                    for term in vocab_translated:
                        if term in embd_dict:
                            ins.append(term)
                        else:
                            not_ins.append(term)
                else:
                    for term in vocab:
                        if term in embd_dict:
                            ins.append(term)
                        else:
                            not_ins.append(term)
                #f.write('OOVs: %s\n' % str(not_ins))
                f.write('OOV Percentage for language %s: %s\n' % (language, (len(not_ins)/len(vocab))))
            f.write('\n')
    f.close()


def main():
    def boolean_string(s):
        if s not in {'False', 'True', 'false', 'true'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True' or s == 'true'
    parser = argparse.ArgumentParser(description='Running XWEAT')
    parser.add_argument('--test_number', type=int, help='Number of the weat test to run', required=False)
    parser.add_argument('--permutation_number', type=int, default=None,
                                            help='Number of permutations (otherwise all will be run)', required=False)
    parser.add_argument('--output_file', type=str, default=None, help='File to store the results)', required=False)
    parser.add_argument('--lower', type=boolean_string, default=False, help='Whether to lower the vocab', required=True)
    parser.add_argument('--similarity_type', type=str, default='cosine', help='Which similarity function to use',
                                            required=False)
    parser.add_argument('--embedding_vocab', type=str, help='Vocab of the embeddings')
    parser.add_argument('--embedding_vectors', type=str, help='Vectors of the embeddings')
    parser.add_argument('--use_glove', type=boolean_string, default=False, help='Use glove')
    parser.add_argument('--postspec', type=boolean_string, default=False, help='Use postspecialized fasttext')
    parser.add_argument('--is_vec_format', type=boolean_string, default=False,
                        help='Whether embeddings are in vec format')
    parser.add_argument('--embeddings', type=str, help='Vectors and vocab of the embeddings')
    parser.add_argument('--lang', type=str, default='en', help='Language to test')
    parser.add_argument('--gender', type=str, default='both', help="Gender settings: 'both', 'female', 'male'")
    parser.add_argument('--word_list_dir', type='str', help='Path to word list files.')
    args = parser.parse_args()

    start = time()
    logging.basicConfig(level=logging.INFO)
    logging.info('XWEAT started')
    weat = XWEAT(gender=args.gender, word_list_dir=args.word_list_dir)
    if args.test_number == 1:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_1()
    elif args.test_number == 2:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_2()
    elif args.test_number == 3:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_3()
    elif args.test_number == 4:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_4()
    elif args.test_number == 5:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_5()
    elif args.test_number == 6:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_6()
    elif args.test_number == 7:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_7()
    elif args.test_number == 8:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_8()
    elif args.test_number == 9:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_9()
    elif args.test_number == 10:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_10()
    elif args.test_number == 11:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_11()
    elif args.test_number == 12:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_12()
    elif args.test_number == 13:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_13()
    elif args.test_number == 14:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_14()
    elif args.test_number == 15:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_15()
    elif args.test_number == 16:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_16()
    elif args.test_number == 17:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_17()
    elif args.test_number == 18:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_18()
    elif args.test_number == 19:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_19()
    elif args.test_number == 20:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_20()
    elif args.test_number == 21:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_21()
    elif args.test_number == 22:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_22()
    elif args.test_number == 23:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_23()
    elif args.test_number == 24:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_24()
    elif args.test_number == 25:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_25()
    elif args.test_number == 26:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_26()
    elif args.test_number == 27:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_27()
    elif args.test_number == 28:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_28()
    elif args.test_number == 29:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_29()
    elif args.test_number == 30:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_30()
    elif args.test_number == 31:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_31()
    elif args.test_number == 32:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_32()
    elif args.test_number == 33:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_33()
    elif args.test_number == 34:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_34()
    elif args.test_number == 35:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_35()
    elif args.test_number == 36:
        targets_1, targets_2, attributes_1, attributes_2 = weat.weat_36()
    else:
        raise ValueError('Only WEAT 1 to 35 are supported')
    if args.lang != 'en':
        logging.info('Translating terms from en to %s', args.lang)
        translation_dict = load_vocab_goran('./data/vocab_dict_en_' + args.lang + '.p')
        targets_1 = translate(translation_dict, targets_1)
        targets_2 = translate(translation_dict, targets_2)
        attributes_1 = translate(translation_dict, attributes_1)
        attributes_2 = translate(translation_dict, attributes_2)
    if args.lower:
        targets_1 = [t.lower() for t in targets_1]
        targets_2 = [t.lower() for t in targets_2]
        attributes_1 = [a.lower() for a in attributes_1]
        attributes_2 = [a.lower() for a in attributes_2]

    if args.use_glove:
        logging.info('Using glove')
        embd_dict = load_embedding_dict(glove=True)
    elif args.postspec:
        logging.info('Using postspecialized embeddings')
        embd_dict = load_embedding_dict(postspec=True)
    elif args.is_vec_format:
        logging.info('Embeddings are in vec format')
        t = time()
        embd_dict = load_embedding_dict(embeddings_path=args.embeddings, glove=False)
        logging.info(f'Loading of embeddings took {round((time() - t) / 60, 2) }')
    else:
        embd_dict = load_embedding_dict(vocab_path=args.embedding_vocab, vector_path=args.embedding_vectors,
                                        glove=False)
    weat.set_embd_dict(embd_dict)

    logging.info('Embeddings loaded')
    logging.info('Running test')
    result = weat.run_test_precomputed_sims(targets_1, targets_2, attributes_1, attributes_2, args.permutation_number,
                                            args.similarity_type)
    results_repr = 'test-statistic: {.3f}, effect-size: {.3f}, p-value: {.3f}'.format(result[0], result[1], result[2])
    logging.info(results_repr)
    mode = 'a' if os.path.exists(args.output_file) else 'w'
    with codecs.open(args.output_file, mode, 'utf8') as f:
        f.write('-----\n')
        f.write('Config: ')
        f.write(str(args.test_number) + ' and ')
        f.write(str(args.lower) + ' and ')
        f.write(str(args.permutation_number) + '\n')
        f.write('Result: ')
        f.write(results_repr)
        f.write(str(result))
        f.write('\n')
        end = time()
        duration_in_hours = ((end - start) / 60) / 60
        f.write(str(duration_in_hours))
        f.write('\n-----\n')
        f.close()

if __name__ == '__main__':
    main()
