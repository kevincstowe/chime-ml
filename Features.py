####
#FEATURES Functions
#All methods of the Features class
####
#try:
#    import theano
#except:
#    pass

from gensim.models import Word2Vec

from operator import add

import math
import operator
import sys
import time
import re
import bz2

import Tools

PMI = True
tbs = False

W2V_MODEL = "models/basic_w2v"
    
def ner(tagged_words):
    print (tagged_words)
    return []
    
def vieweg(tweet_data):
    return []

w2v=None
def cbow_feature(tweet, average=False):
    global w2v
    if not w2v:
        w2v = Word2Vec.load(W2V_MODEL)

    res = [0]*200

    for word in tweet:
        if word in w2v:
            res = map(add, res, w2v[word])

    ##Implement average cbow##
            
    return list(res)

def rt_feature(tweet):
    res = [0, 0]
    if len(tweet) > 0:
        if tweet[0].lower() == "rt":
            res[0] += 1
        if "rt" in tweet[1:]:
            res[1] += 1
    return res

def web_feature(tweet):
    all_webs = sorted([line.strip() for line in open("resources/all_webs.txt")])
    for word in tweet:
        if word in all_webs:
            return [1]

    return [0]

def generate_buckets(number_of_buckets = 50, start_date = "2012-10-23 00:00:00", end_date="2012-11-09 00:00:00"):
    start_time = time.mktime(time.strptime(start_date, "%Y-%m-%d %H:%M:%S"))
    end_time = time.mktime(time.strptime(end_date, "%Y-%m-%d %H:%M:%S"))
    buckets = [start_time]

    interval_size = (end_time - start_time) / number_of_buckets

    for i in range(number_of_buckets):
        start_time += interval_size
        buckets.append(start_time)

    return buckets

buckets = None
def time_bucket(date, number_of_buckets=0, one_hot=True):  
    global buckets
    if not buckets:
        buckets = generate_buckets(number_of_buckets)
    if not date:
        if one_hot:
            return [0]*len(buckets)
        else:
            return [0]
    try:            
        date_time = time.mktime(time.strptime(date, "%Y-%m-%d %H:%M:%S"))

        for i in range(len(buckets)):
            if date_time >= buckets[i] and date_time < buckets[i+1]:
                if one_hot:
                    result = [0]*len(buckets)
                    result[i] = 1
                    return result
                else:
                    return [i+1]

        if date_time < buckets[0]:
            return [1] + [0]*(len(buckets)-1)
        elif date_time >= buckets[len(buckets)-1]:
            return [0] * (len(buckets)-1) + [1]
        else:
            print ("Bucketing failed. Datetime : " + str(date_time))
            sys.exit(2)
    except Exception as e:
        print ("Bucketing failed...!")
        print ("Exception : " + str(e))
        print ("Attempted date_time : " + str(date_time))
        print ("Returning zero vector")
        if one_hot:
            return [0]*len(buckets)
        else:
            return [0]       

def build_structures(training_data, tagged_data=None, key_term_count=0, bow_count=0, bigram_count=0, trigram_count=0, add_pos=False, tag="None"):
    result = {}

    if key_term_count > 0:
        result["key_term_dict"] = build_kt_dict(training_data, number_of_terms=key_term_count, min_count=3, gram=0, pos=False, tag=tag)
        if add_pos:
            result["tagged_key_term_dict"] = build_kt_dict(tagged_data, number_of_terms=key_term_count, gram=0, pos=True)
    if bow_count > 0:
        result["bow_dict"] = build_bow_dict(training_data, bow_count)
    if bigram_count > 0:
        result["bigram_dict"] = build_kt_dict(training_data, number_of_terms=bigram_count, gram=1, pos=False, tag=tag)
        if add_pos:
            result["tagged_bigram_dict"] = build_kt_dict(tagged_data, number_of_terms=bigram_count, gram=1, pos=True)
    if trigram_count > 0:
        result["trigram_dict"] = build_kt_dict(training_data, number_of_terms=trigram_count, gram=2, pos=False, tag=tag)
        if add_pos:
            result["tagged_trigram_dict"] = build_kt_dict(tagged_data, number_of_terms=trigram_count, gram=2, pos=True)

    return result

def build_bow_dict(data, min_count=10, w2v=False, n=5):
    d = {}
    try:    
        count = 0
        for key in data.keys():
            count += 1
            for word in data[key]["words"]:
                Tools.add_count_to_dict(d, word)
    except AttributeError as e:
        for line in data:
            for word in Tools.normalize_string(line):
                Tools.add_count_to_dict(d, word)

    res = []
    for key in d.keys():
        if d[key] >= min_count:
            res.append(key)

    return sorted(res)

def pull_key_terms(d, min_count):
    kt_vals = {}
    for key in [k for k in d.keys() if sum(d[k]) >= min_count]:
        #negative term is zero, take the pos term
        if d[key][0] == 0 and d[key][1] > min_count:
            kt_vals[key] = float(d[key][1])
        #pos term is zero, take the neg term
        elif d[key][1] == 0 and d[key][0] > min_count:
            kt_vals[key] = -float(d[key][0])
        #both terms non zero, take the higher one
        elif d[key][0] > 0 and d[key][1] > 0:
            if d[key][1] > d[key][0]:
                kt_vals[key] = float(max(d[key][0], d[key][1])) / float(min(d[key][0], d[key][1]))
            else:
                kt_vals[key] = -float(max(d[key][0], d[key][1])) / float(min(d[key][0], d[key][1]))
            #something failed : likely both below min count. take 0
        else:
            kt_vals[key] = 0.
    return kt_vals

def build_kt_dict(data, gram=0, number_of_terms=500, min_count=0, pos=False, tag="None"):
    d = {}
    tot_pos = 0.
    tot_neg = 0.
    tot_all = 0.

    for key in data.keys():
        if pos:
            word_data = [data[key]["words"][i] + "/" + data[key]["pos_tags"][i] for i in range(len(data[key]["words"]))]
        else:
            word_data = data[key]["words"]

        tot_all += 1
        anns = data[key]["annotations"]

        for i in range(len(word_data)-(gram)):
            if gram > 0:
                word = "_".join(word_data[i:i+gram+1])
            else:
                word = word_data[i]
                    
            #sloppy but works : if the words not in the dictionary, generate it as a two element array, 0:negatives, 1:positives, and incremenet as they're seen
            if word not in d:
                if "None" in anns or (tag != "None" and tag not in anns and tag not in [t.split("-")[0] for t in anns]):
                    tot_neg += 1
                    d[word] = [1, 0]
                else:
                    tot_pos += 1
                    d[word] = [0, 1]
            else:
                if "None" in anns or (tag != "None" and tag not in anns and tag not in [t.split("-")[0] for t in anns]):
                    tot_neg += 1
                    d[word][0] += 1
                else:
                    tot_pos += 1
                    d[word][1] += 1

    #calculate PMI if it's called, tends to work better than counts
    if PMI:
        pmi_pos = {}
        pmi_neg = {}
        for word in [w for w in d.keys() if sum(d[w]) >= min_count]:
            if d[word][1] > 0:
                pmi_pos[word] = math.log((d[word][1]/tot_all)/((sum(d[word])/tot_all)*(tot_pos/tot_all)))
            if d[word][0] > 0:
                pmi_neg[word] = math.log((d[word][0]/tot_all)/((sum(d[word])/tot_all)*(tot_neg/tot_all)))
        #wacky list comprehension, gets the bottom (highest pmi) number_of_terms items off the PMI dict
        result = [item[0] for item in sorted(pmi_pos.items(), key=operator.itemgetter(1), reverse=True)][:number_of_terms]
    else:
        d = pull_key_terms(d, min_count)
        #two list comprehensions : one gets bottom number_of_terms/2, one gets top number_of_terms/2, which yields number_of_items terms with highest and lowest counts
        result = [item[0] for item in sorted(d.items(), key=operator.itemgetter(1), reverse=True)][:number_of_terms/2] + [item[0] for item in sorted(d.items(), key=operator.itemgetter(1))][:number_of_terms/2] 

    return result

def bow_features(words, tagged_words=[], data_structures=None):
    features = []
    if "key_term_dict" in data_structures.keys():
        features += bow(words, data_structures["key_term_dict"], gram=1)
    if "bow_dict" in data_structures.keys():
        features += bow(words, data_structures["bow_dict"], gram=1)
    if "bigram_dict" in data_structures.keys():
        features += bow(words, data_structures["bigram_dict"], gram=2)
    if "trigram_dict" in data_structures.keys():
        features += bow(words, data_structures["trigram_dict"], gram=3)

    if "tagged_key_term_dict" in data_structures.keys():
        features += bow(tagged_words, data_structures["tagged_key_term_dict"])
    if "tagged_bigram_dict" in data_structures.keys():
        features += bow(tagged_words, data_structures["tagged_bigram_dict"], gram=2)
    if "tagged_trigram_dict" in data_structures.keys():
        features += bow(tagged_words, data_structures["tagged_trigram_dict"], gram=3)

    return features

def bow(tweet, d, gram=1):
    result = [0] * len(d)

    for i in range(len(tweet)-(gram-1)):
        if gram > 1:
            word = "_".join(tweet[i:i+gram])
        else:
            word = tweet[i]
        if word in d:
            result[d.index(word)] += 1

    return result
