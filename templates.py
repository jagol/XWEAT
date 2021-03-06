# Templates from May et al. 2019: "On Measuring Bias in Sentence Encoders"
# File: https://github.com/W4ngatang/sent-bias/blob/master/scripts/generate_basic_contexts.py


from weat import XWEAT


NAME_TEMPLATES = (
    'Das ist {term}.',  # This is/That is
    'Dort ist {term}.',  # There is
    'Hier ist {term}.',  # Here is
    '{term} ist hier.',  # is here
    '{term} ist dort.',  # is there
    '{term} ist eine Person.',  # is a person
    'Der Name der Person ist {term}.',  # The person\'s name is
)


SUBJECT_PRONOUN_TEMPLATES = (
    '{term} is here.',
    '{term} is there.',
    'Here {term} is.',
    'There {term} is.',
    '{term} is a person.',
)


OBJECT_PRONOUN_TEMPLATES = (
    'It is {term}.',
    'This is {term}.',
    'That is {term}.',
)


POSSESSIVE_PRONOUN_TEMPLATES = (
    'This is {term}.',
    'That is {term}.',
    'There is {term}.',
    'Here is {term}.',
    'It is {term}.',
    '{term} is there.',
    '{term} is here.',
)


MASS_NOUN_TEMPLATES = (
    'This is {term}.',
    'That is {term}.',
    'There is {term}.',
    'It is {term}.',
)


SINGULAR_NOUN_TEMPLATES = (
    'This is {article} {term}.',
    'That is {article} {term}.',
    'There is {article} {term}.',
    'Here is {article} {term}.',
    'The {term} is here.',
    'The {term} is there.',
)


PLURAL_NOUN_TEMPLATES = (
    'These are {term}.',
    'Those are {term}.',
    'They are {term}.',
    'The {term} are here.',
    'The {term} are there.',
)


class Template:

    def __init__(self, templates):
        self.templates = templates  # List of tuples of templates

    def fill_templates(self, term):
        filled_templates = []
        for tp in self.templates:
            for t in tp:
                filled_templates.append(t.format(term=term))
        return filled_templates

    def fill_templates_mul_terms(self, terms):
        filled_templates = []
        for term in terms:
            filled_templates.extend(self.fill_templates(term))
        return filled_templates


def create_filled_template_from_word_list(fpath_in, fpath_out, ttype='name'):
    """Given a word-list file and a type (names, words) fill in templates and produce output file."""
    mapping = {
        'name': [NAME_TEMPLATES],
        'mname': [NAME_TEMPLATES],
        'fname': [NAME_TEMPLATES]
    }
    if ttype == 'name':
        terms = [s.capitalize() for s in XWEAT.load_names(fpath_in, shuffle=False)]
    elif ttype == 'mname':
        terms = [s.capitalize() for s in XWEAT.load_male_names(fpath_in, shuffle=False)]
    elif ttype == 'fname':
        terms = [s.capitalize() for s in XWEAT.load_female_names(fpath_in, shuffle=False)]
    elif ttype == 'words':
        terms = XWEAT.load_word_list(fpath_in)
    else:
        raise Exception(f'ttype {ttype} is not known.')
    filled_templates = Template(mapping[ttype]).fill_templates_mul_terms(terms)
    with open(fpath_out, 'w') as fout:
        for ft in filled_templates:
            fout.write(ft + '\n')


if __name__ == __name__:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--fpath_in', required=True, help='Path to input file: one name or word per line.')
    parser.add_argument('-o', '--fpath_out', required=True, help='Path to output file: one sentence per line.')
    parser.add_argument('-t', '--ttype', required=True, help="Either 'name', 'mname', 'fname' or 'word'.")
    args = parser.parse_args()
    create_filled_template_from_word_list(fpath_in=args.fpath_in, fpath_out=args.fpath_out, ttype=args.ttype)
