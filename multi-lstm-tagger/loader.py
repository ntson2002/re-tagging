import os
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
from utils import iob2, iob_iobes


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        # print s

        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            print i, s_str
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')



def update_tag_scheme_multilayer(sentences, gold_columns, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    def iob2_iobes(tags):
        new_tags = []
        for i, tag in enumerate(tags):
            if tag == 'O':
                new_tags.append(tag)
            else:
                split = tag.split('-')
                if split[0] == "I":
                    j = i + 1
                    while j < len(tags) and tags[j] == "O":
                        j = j + 1
                    next_tag = ""
                    if j < len(tags):
                        next_tag = tags[j].split("-")[0]

                    if next_tag == "" or next_tag == "B":
                        new_tags.append(tag.replace('I-', 'E-'))
                    else:
                        new_tags.append(tag)
                else:  # split[0] == 'B':
                    if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                        new_tags.append(tag)
                    else:
                        new_tags.append(tag) # giu nguyen tag B, khong thay thanh tag S vi corpus co it qua !

                        # # new_tags.append(tag.replace('B-', 'S-')) # Sua ngay 3/12
                        # j = i + 1
                        # while j < len(tags) and tags[j] == "O":
                        #     j = j + 1
                        # next_tag = ""
                        # if j < len(tags):
                        #     next_tag = tags[j].split("-")[0]
                        #
                        # if next_tag == "" or next_tag == "B":
                        #     new_tags.append(tag.replace('B-', 'S-'))
                        # else:
                        #     new_tags.append(tag)

        return new_tags

    def check_iob2(tags):
        """
        Check that tags have a valid IOB2 format.
        """
        for i, tag in enumerate(tags):
            if tag == 'O':
                continue
            split = tag.split('-')
            if len(split) != 2 or split[0] not in ['I', 'B']:
                return False
        return True


    for i, s in enumerate(sentences):
        # print s
        tags_atlayers = {c:[w[c] for w in s] for c in gold_columns}

        for c in gold_columns:
            tags = tags_atlayers[c]
            # Check that tags are given in the IOB format
            if not check_iob2(tags):
                s_str = '\n'.join(' '.join(w) for w in s)
                print i, s_str
                raise Exception('Sentences should be given in IOB format! ' +
                                'Please check sentence %i:\n%s' % (i, s_str))


            if tag_scheme == 'iobes':
                new_tags = iob2_iobes(tags)
                for word, new_tag in zip(s, new_tags):
                    word[c] = new_tag
            else:
                raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print "Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    )
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print "Found %i unique characters" % len(dico)
    return dico, char_to_id, id_to_char


def tag_mapping(sentences, column):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    # tags = [[word[-1] for word in s] for s in sentences]

    tags = [[word[column] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)

    print "Found %i unique named entity tags" % len(dico)

    return dico, tag_to_id, id_to_tag


def pos_mapping(sentences, col=1):
    """
    Create a dictionary and a mapping of pos tags, sorted by frequency.
    """
    tags = [[word[col] for word in s] for s in sentences]
    dico = create_dico(tags)
    pos_to_id, id_to_pos = create_mapping(dico)

    print "Found %i unique POS tags" % len(dico)

    return dico, pos_to_id, id_to_pos


def chunk_mapping(sentences, col=2):
    """
    Create a dictionary and a mapping of chunk tags, sorted by frequency.
    """
    tags = [[word[col] for word in s] for s in sentences]
    dico = create_dico(tags)
    chunk_to_id, id_to_chunk = create_mapping(dico)
    print "Found %i unique Chunk tags" % len(dico)
    return dico, chunk_to_id, id_to_chunk


def feature_mapping(sentences, feature={'name': 'pos_dim', 'column': 1, 'dim': 5}):
    """
    Create a dictionary and a mapping of pos tags, sorted by frequency.
    """
    col = feature['column']
    name = feature['name']
    tags = [[word[col] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico['<UNK>'] = 0
    tag_to_id, id_to_tag = create_mapping(dico)

    print "Found %i unique %s tags" % (len(dico), name)

    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """

    def f(x): return x.lower() if lower else x

    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }


# def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, pos_to_id, pos_col=1, chunk_to_id=None, chunk_col=2, lower=False):
#     """
#     Prepare the dataset. Return a list of lists of dictionaries containing:
#         - word indexes
#         - word char indexes
#         - tag indexes
#     """
#     def f(x): return x.lower() if lower else x
#     data = []
#     for s in sentences:
#         str_words = [w[0] for w in s]
#         words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
#                  for w in str_words]
#         # Skip characters that are not in the training set
#         chars = [[char_to_id[c] for c in w if c in char_to_id]
#                  for w in str_words]
#         caps = [cap_feature(w) for w in str_words]
#
#         tags = [tag_to_id[w[-1]] for w in s]
#
#         poss = [pos_to_id[w[pos_col]] for w in s]
#
#         chunks = [chunk_to_id[w[chunk_col]] for w in s]
#
#         data.append({
#             'str_words': str_words,
#             'words': words,
#             'chars': chars,
#             'caps': caps,
#             'tags': tags,
#             'poss': poss,
#             'chunks' : chunks
#         })
#     return data
#
#
# def prepare_dataset2(sentences, word_to_id, char_to_id, tag_to_id, feature_maps, lower):
#
#     """
#     Prepare the dataset. Return a list of lists of dictionaries containing:
#         - word indexes
#         - word char indexes
#         - tag indexes
#     """
#
#     def f(x): return x.lower() if lower else x
#     data = []
#     for s in sentences:
#         str_words = [w[0] for w in s]
#         words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
#                  for w in str_words]
#         # Skip characters that are not in the training set
#         chars = [[char_to_id[c] for c in w if c in char_to_id]
#                  for w in str_words]
#         caps = [cap_feature(w) for w in str_words]
#
#         tags = [tag_to_id[w[-1]] for w in s]
#
#         data_object = {
#             'str_words': str_words,
#             'words': words,
#             'chars': chars,
#             'caps': caps,
#             'tags': tags
#         }
#
#         for fm in feature_maps:
#             ftag_to_id = [fm['ftag_to_id'][w[fm['column']]] for w in s]
#             data_object[fm['name']] = ftag_to_id
#         data.append(data_object)
#     return data


def prepare_dataset3(sentences, word_to_id, char_to_id, tag_maps, feature_maps, lower):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    def f(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]

        # tags = [tag_to_id[w[-1]] for w in s]

        data_object = {
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps
            # 'tags': tags
        }

        for tm in tag_maps:
            # tags = [tm['tag_to_id'][w[tm['column']]] for w in s]

            # update if tag not appear in training data --> convert into id of tag "O"
            tags = [tm['tag_to_id'][w[tm['column']]] if w[tm['column']] in tm['tag_to_id'] else tm['tag_to_id']["O"] for w in s]

            # print tm
            data_object['tags' + str(tm['layer'])] = tags
            # print [w[3] for w in s]
            # print tags
            # assert 0


        for fm in feature_maps:
            try:
                ftag_to_id = [fm['ftag_to_id'][w[fm['column']] if w[fm['column']] in fm['ftag_to_id'] else '<UNK>'] for w in s]
                # ftag_to_id = [fm['ftag_to_id'][w[fm['column']]] for w in s]
            except Exception as e:
                from pprint import pprint
                pprint(fm)
                print e

            data_object[fm['name']] = ftag_to_id

        data.append(data_object)

    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """

    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
                         line.rstrip().split()[0].strip()
                         for line in codecs.open(ext_emb_path, 'r', 'utf-8')
                         if len(ext_emb_path) > 0
                         ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word
