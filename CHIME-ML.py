import json
import random

import Features
import Learn
import Tools 

PART = 1

BEST_FEATS = [0, 1000, 0, 3000, 3000, 2, 384, True, True, True, True]
UNI_FEATS = [0, 200, 0, 0, 0, 0, 0, False, False, False, False]

FEATS = UNI_FEATS

BOW_BEST = FEATS[0]
KEY_TERM_BEST = FEATS[1]
#? = FEATS[2]
BIGRAM_BEST = FEATS[3]
TRIGRAM_BEST = FEATS[4]
CONTEXT_SIZE = FEATS[5]
BUCKETS_BEST = FEATS[6]
RTS = FEATS[7]
WEB = FEATS[8]
POS = True #FEATS[9]
W2V = FEATS[10]

NER = False
VIEWEG = False

LDA = False

if PART == 1:
    TRAIN_DATA = "data/part1/part1_tagged.json"
else:
    TRAIN_DATA = "data/part2/part2_tagged.json"
    
def run_cv(cat="None", param=None, splits=5):
    def pos_json(json_data):
        result = {}
        for key in json_data.keys():
            result[key] = json_data[key]
            result[key]["text"] = " ".join(["/".join(word.split("/")[0:-3]) + "/" + word.split("/")[-3] + "/" +  word.split("/")[-2] for word in json_data[key]["text"].split()])
        return result

    train_json = pos_json(json.load(open(TRAIN_DATA)))
    train_keys = list(train_json.keys())

    f1 = 0
    prec = 0
    rec = 0

    for i in range(splits):
        cv_test_json = {key:train_json[key] for key in train_keys[int(i*len(train_keys)/5):int((i+1)*len(train_keys)/5)]}
        cv_train_json = {key:train_json[key] for key in train_keys if key not in cv_test_json.keys()}

        data_structures = Features.build_structures(cv_train_json, tagged_data=train_json, key_term_count=KEY_TERM_BEST, bow_threshhold=BOW_BEST, bigram_threshhold=BIGRAM_BEST, trigram_threshhold=TRIGRAM_BEST, add_pos=POS, tag=cat)

        train_data, test_data = vectorize_json(cv_train_json, cv_test_json, cat, data_structures)

        if param:
            res = Learn.learn(train_data, test_data, mod="SVM", keys=True, param=param)
        else:
            res = Learn.learn(train_data, test_data, mod="SVM", keys=True)

        j = res[0]
        scores = res[1]
        
        f1 += scores[0]
        prec += scores[1]
        rec += scores[2]
    return (f1/splits, prec/splits, rec/splits)

def vectorize_json(train_j, test_j, category, data_structures, tagged=True, context_data=None):
    count = 0
    all_data = json.load(open(TRAIN_DATA))
    all_data = {str(int(float(k))) : all_data[k] for k in all_data.keys()}
#    for k in all_data.keys():
#        all_data[k]["id"] = str(int(float(all_data[k]["id"])))
        
    train_test = []
    for j in (train_j, test_j):
        current_vectors = {}
        for key in j.keys():
            tagged_words = []
            if tagged:
                tags = [w.split("/")[-1] for w in j[key]["text"].split()]
                sent_string = " ".join(["/".join(w.split("/")[:-2]) for w in j[key]["text"].split()])
                tagged_words = [Tools.normalize_word(sent_string.split()[i]) + "/" + tags[i] for i in range(len(sent_string.split()))]
            else:
                sent_string = j[key]["text"]

            reg_words = [Tools.normalize_word(w) for w in sent_string.split()]
            features = Features.featurize(reg_words, tagged_words=tagged_words, data_structures=data_structures)
            print (sent_string, reg_words, tagged_words)

            if not all([v == 0 for v in features]):
                print (reg_words, tagged_words)
                print (Features.featurize(reg_words, tagged_words=tagged_words, data_structures=data_structures))
                print (data_structures['key_term_dict'])
                print (data_structures['tagged_key_term_dict'])

            if BUCKETS_BEST > 0:
                features += Features.time_bucket(j[key]["date"], number_of_buckets=BUCKETS_BEST)
            if RTS:
                features += Features.rt_feature(reg_words)
            if WEB:
                features += Features.web_feature(reg_words)
            if W2V:
                features += Features.cbow_feature(reg_words)
    
            if NER:
                features += Features.ner(tagged_words)
            if VIEWEG:
                features += Features.vieweg(j[key])
                
            if CONTEXT_SIZE > 0:
                for i in range(1, CONTEXT_SIZE+1):
                    if not context_data:
                        context_data = all_data
                    active_key = Tools.find_prev_key(context_data, key, context=i)
                    if active_key:
                        sent_words = Tools.normalize_string(context_data[active_key]["text"])
                    else:
                        sent_words = []

                    features += Features.featurize(sent_words,data_structures=data_structures)

            if category != "None":
                if category in j[key]["annotations"] or category in [a.split("-")[0] for a in j[key]["annotations"]]:
                    features.append(1)
                else:
                    features.append(0)
            else:
                if "relevant-no" not in j[key]["annotations"] and "None" not in j[key]["annotations"]:
                    features.append(1)
                else:
                    features.append(0)
            current_vectors[key] = features
        train_test.append(current_vectors)
    return train_test

for p in [2500]:
    print (p)
    print (run_cv(param=p))
