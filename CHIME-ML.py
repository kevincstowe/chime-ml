import json
import random

import Features
import Learn
import Tools 

PART = 1

BEST_FEATS = [0, 1000, 3000, 3000, 2, 384, True, True, True, True, True, True]
UNI_FEATS = [0, 200, 0, 0, 0, 0, False, False, False, False, False, False]

FEATS = UNI_FEATS

BOW_BEST = FEATS[0]
KEY_TERM_BEST = FEATS[1]
BIGRAM_BEST = FEATS[2]
TRIGRAM_BEST = FEATS[3]
CONTEXT_SIZE = FEATS[4]
BUCKETS_BEST = FEATS[5]
RTS = FEATS[6]
WEB = FEATS[7]
POS = FEATS[8]
W2V = FEATS[9]

NER = FEATS[10]
VIEWEG = FEATS[11]

ALGORITHM = "SVM"

if PART == 1:
    DATA = "data/part1/part1_tagged.json"
else:
    DATA = "data/part2/part2_tagged.json"

def build_pos_json(json_data):
    for key in json_data.keys():
        if not json_data[key]["text"]:
            json_data[key]["pos_tags"] = []
            json_data[key]["ner_tags"] = []
            json_data[key]["words"] = []
            json_data[key]["tagged_words"] = []
        else:
            pos_tags, ner_tags, words, tagged_words = zip(*[(w.split("/")[-2], w.split("/")[-3], Tools.normalize_word("/".join(w.split("/")\
[:-3])), Tools.normalize_word("/".join(w.split("/")[:-3])) + "/" + w.split("/")[-2]) for w in json_data[key]["text"].split()])
            json_data[key]["pos_tags"] = pos_tags
            json_data[key]["ner_tags"] = ner_tags
            json_data[key]["words"] = words
            json_data[key]["tagged_words"] = tagged_words
    return json_data
    
def run_cv(cat="None", param=None, splits=5):
    data_json = build_pos_json(json.load(open(DATA)))
    data_keys = list(data_json.keys())

    f1 = 0
    prec = 0
    rec = 0

    for i in range(splits):
        cv_test_json = {key:data_json[key] for key in data_keys[int(i*len(data_keys)/5):int((i+1)*len(data_keys)/5)]}
        cv_train_json = {key:data_json[key] for key in data_keys if key not in cv_test_json.keys()}

        data_structures = Features.build_structures(cv_train_json, tagged_data=data_json, key_term_count=KEY_TERM_BEST, bow_count=BOW_BEST, bigram_count=BIGRAM_BEST, trigram_count=TRIGRAM_BEST, add_pos=POS, tag=cat)

        train_data = vectorize_json(cv_train_json, cat, data_structures, context_data=data_json)
        test_data = vectorize_json(cv_test_json, cat, data_structures, context_data=data_json)

        if param:
            res = Learn.learn(train_data, test_data, mod=ALGORITHM, keys=True, param=param)
        else:
            res = Learn.learn(train_data, test_data, mod=ALGORITHM, keys=True)

        j = res[0]
        scores = res[1]
        
        f1 += scores[0]
        prec += scores[1]
        rec += scores[2]
    return (f1/splits, prec/splits, rec/splits)

def vectorize_json(js, category, data_structures, tagged=True, context_data=None):
    count = 0
#    for k in all_data.keys():
#        all_data[k]["id"] = str(int(float(all_data[k]["id"])))
        
    current_vectors = {}
    for key in js.keys():
        features = Features.bow_features(js[key]["words"], tagged_words=js[key]["tagged_words"], data_structures=data_structures)
        
        if BUCKETS_BEST > 0:
            features += Features.time_bucket(js[key]["date"], number_of_buckets=BUCKETS_BEST)
        if RTS:
            features += Features.rt_feature(js[key]["words"])
        if WEB:
            features += Features.web_feature(js[key]["words"])
        if W2V:
            features += Features.cbow_feature(js[key]["words"])
    
        if NER:
            features += Features.ner(js[key]["ner_tags"])
        if VIEWEG:
            features += Features.vieweg(j[key])
                
        if CONTEXT_SIZE > 0:
            for i in range(-CONTEXT_SIZE, CONTEXT_SIZE+1):
                if not context_data:
                    return []

                active_key = Tools.find_key(key, context_data, context=i)
                if active_key:
                    sent_words = context_data[active_key]["words"]
                else:
                    sent_words = []

                features += Features.bow_features(sent_words,data_structures=data_structures)

        if category != "None":
            if category in js[key]["annotations"] or category in [a.split("-")[0] for a in js[key]["annotations"]]:
                features.append(1)
            else:
                features.append(0)
        else:
            if "relevant-no" not in js[key]["annotations"] and "None" not in js[key]["annotations"]:
                features.append(1)
            else:
                features.append(0)
        current_vectors[key] = features
    return current_vectors

print (run_cv(param=p))
