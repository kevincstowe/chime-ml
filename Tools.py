####
# TOOLS
# All sorts of helper functions
####

import string
import re
import operator

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import tokenize
from nltk.stem.porter import PorterStemmer

import datetime

web_pattern = None
num_pattern = None
stemmer = None
tokenizer = None
all_stopwords = None

EXTRA_STOPWORDS = ["ll", "rt"]

def xldate_to_datetime(xldate):
    temp = datetime.datetime(1899, 12, 30)
    delta = datetime.timedelta(days=xldate)
    return temp+delta

def find_prev_key(all_data, current_key, context):
    if context == 0:
        if current_key and current_key != "None":
            try:
                return all_data[current_key]["id"]
            except KeyError as e:
                return ""
        else:
            return ""
    else:
        if current_key in all_data.keys():
            return find_prev_key(all_data, all_data[current_key]["previous"], context-1)
        else:
            return ""

def weighted_word_frequencies(doc_json, words=10, normalize=True):
    pos_wfs = {}
    all_wfs = {}

    for document in doc_json.keys():
        for word in Set(normalize_string(doc_json[document]["text"])):
            if "None" not in doc_json[document]["annotations"]:
                Tools.add_count_to_dict(pos_wfs, word)
            Tools.add_count_to_dict(all_wfs, word)

    final_wfs = {}
    for key in pos_wfs:
        if all_wfs[key] > 3 and float(pos_wfs[key])/all_wfs[key] > .5:
            final_wfs[key] = float(pos_wfs[key])/all_wfs[key]

    return sorted(final_wfs.items(), key=operator.itemgetter(1), reverse=True)[:words]


def add_count_to_dict(d, item):
    if item not in d:
        d[item] = 1
    else:
        d[item] += 1

def tokenize_sent(text):
    tokenizer = tokenize.casual.TweetTokenizer()
    return tokenizer.tokenize(text)

def normalize_word(w, extras=True, stem=False):
    global web_pattern
    global num_pattern
    global total_webs
    def compress(w):
        for c in set(w):  
            try:
                pattern = re.compile("[" + str(c) + "]{3,}")
            except:
                return w #something else needs to be working here...

        w = re.sub(pattern, c, w)
        return w

    if stem:
        global stemmer
        if not stemmer:
            stemmer = PorterStemmer()
        if "#" not in w and "@" not in w:
            w = stemmer.stem(w)

    if not web_pattern:
        web_pattern = re.compile(r".*www\.|http:|https:.*")
    if not num_pattern:
        num_pattern = re.compile("[0-9]+")

    w = compress(w).lower()

    if not extras and not ([c for c in w if c not in string.punctuation] or num_pattern.match(w) or web_pattern.match(w)):
        return ""

    if not [c for c in w if c not in string.punctuation]:
        w = "[punct]"
    if num_pattern.match(w):
        w = "[num]"
    elif web_pattern.match(w):
#        w = "[web]"
        try:
            w = w.split("://")[1].split("/")[0]
        except IndexError as e:
            print ("!! " + w)
    elif w[0] == "@":
        w = "[user]"

    return w

def normalize_string(s, remove_stopwords=True, tokenize=True, extras=True, stem=False):
    def encoding_norm(l):
        new_l = []
        for w in l:
            try:
                w.encode("UTF-8")
                new_l.append(w)
            except:
                pass
        return new_l

    if tokenize:
        result = tokenize_sent(s)
        result = normalize_list(result, remove_stopwords=remove_stopwords, extras=extras, stem=stem)
        return encoding_norm(result)
    else:
        return encoding_norm(normalize_list(s.split(), remove_stopwords=remove_stopwords, extras=extras, stem=stem))

def normalize_list(l, remove_stopwords, extras, stem):
    global all_stopwords

    result_list = []
    for i in range(len(l)):
        res = normalize_word(l[i], extras)
        if res:
            result_list.append(res)

    if remove_stopwords:
        if not all_stopwords:
            all_stopwords = stopwords.words('english') + EXTRA_STOPWORDS
        result_list = [w.lower() for w in result_list if w.lower() not in all_stopwords]

    return result_list
