
# CHIME-ML
# Primary application for classification of the Hurricane Sandy data. Loads and featurizes the necessary json, and runs 5-fold CV.
import json
import random
import sys
import getopt

import Features
import Learn
import Tools 

#basic feature sets for playing with
BEST_FEATS = [0, 1000, 3000, 3000, 2, 384, True, True, True, True, True, True]
BASELINE_FEATS = [0, 1000, 0, 0, 0, 0, False, False, False, False, False, False]

#Pick which feature set you'd like, or set each feature manually
FEATS = BASELINE_FEATS

BOW_BEST = FEATS[0]        #how many terms to include using the bag-of-words method. depracated, as it underperforms key terms
KEY_TERM_BEST = FEATS[1]   #how many terms to include,    PMI key term method
BIGRAM_BEST = FEATS[2]     #how many bigrams to include,  PMI key term method
TRIGRAM_BEST = FEATS[3]    #how many trigrams to include, PMI key term method
CONTEXT_SIZE = FEATS[4]    #how many tweets of context to include. currently only adds key term context, but could include others
BUCKETS_BEST = FEATS[5]    #how many buckets to use to represent the time. a value of 0 excludes the time feature
RTS = FEATS[6]             #boolean whether to include RTs as a feature
WEB = FEATS[7]             #boolean whether to include URLs as a feature
POS = FEATS[8]             #boolean whether to include POS tags - will affect key terms, bigrams, and trigrams 
W2V = FEATS[9]             #boolean whether to include word embeddings

NER = FEATS[10]            #NOT IMPLEMENTED - boolean whether to include named entities 
VIEWEG = FEATS[11]         #NOT IMPLEMENTED - boolean whether to include features from Vieweg et al 

ALGORITHM = "SVM"          #Pick your algorithm - currently can be Support Vector Machines ('SVM'), Logistic Regression ('LR'), or Naive Bayes ('NB'). More could be added via the Learn module

#You'll have to change this to a json with non-empty 'text' fields -- see https://github.com/kevincstowe/chime-ml/README.md
DEFAULT_DATA = "data/part1/part1_cleaned.json"

#Method for loading the data. It takes the json, and splits the tagged text into four different useful attributes
def build_pos_json(json_data):
    for key in json_data.keys():
        #TODO : It is weird that this happens
        if not json_data[key]["text"]:
            json_data[key]["pos_tags"] = []
            json_data[key]["ner_tags"] = []
            json_data[key]["words"] = []
            json_data[key]["tagged_words"] = []
        else:
            #One list-comprehension pass through gets all of our data structures
            pos_tags, ner_tags, words, tagged_words = zip(*[(w.split("/")[-2], w.split("/")[-3], Tools.normalize_word("/".join(w.split("/")\
[:-3])), Tools.normalize_word("/".join(w.split("/")[:-3])) + "/" + w.split("/")[-2]) for w in json_data[key]["text"].split()])
            json_data[key]["pos_tags"] = pos_tags
            json_data[key]["ner_tags"] = ner_tags
            json_data[key]["words"] = words
            json_data[key]["tagged_words"] = tagged_words
    return json_data
    
#Heavy lifting. Loads the data, splits it into partitions, and passes it through the Learn module. 
#tag       : Which tag to test
#param     : Parameter that is passed to the machine learning algorithm. Allows for parameter testing
#n         : Number of cross validation folds to go through
def run_cv(data=DEFAULT_DATA, tag="none", param=None, n=5):
    data_json = build_pos_json(json.load(open(data)))
    data_keys = list(data_json.keys())
    print (tag)
    f1 = 0
    prec = 0
    rec = 0

    for i in range(n):
        #get each split as test, then take the rest as training
        cv_test_json = {key:data_json[key] for key in data_keys[int(i*len(data_keys)/n):int((i+1)*len(data_keys)/n)]}
        cv_train_json = {key:data_json[key] for key in data_keys if key not in cv_test_json.keys()}

        #builds all the lexical dictionaries for key terms, bigrams, and trigrams
        #TODO : Does this need 'tagged_data'?? I don't think so
        data_structures = Features.build_structures(cv_train_json, tagged_data=data_json, key_term_count=KEY_TERM_BEST, bow_count=BOW_BEST, bigram_count=BIGRAM_BEST, trigram_count=TRIGRAM_BEST, add_pos=POS, tag=tag)

        #Turn both training and test data into dictionaries of {key:feature_vector}
        #context_data is required - if tweets are randomized, previous and following tweets may not be in the right json
        train_data = vectorize_json(cv_train_json, tag, data_structures, context_data=data_json)
        test_data = vectorize_json(cv_test_json, tag, data_structures, context_data=data_json)

        #pass the vectors to the Learn module, which returns a dictionary of predictions {key:1 or 0}, as well as (f1, prec, rec)
        if param:
            preds, scores = Learn.learn(train_data, test_data, mod=ALGORITHM, keys=True, param=param)
        else:
            preds, scores = Learn.learn(train_data, test_data, mod=ALGORITHM, keys=True)    
        
        f1 += scores[0]
        prec += scores[1]
        rec += scores[2]

    #after n passes, average the results
    return (f1/n, prec/n, rec/n)

def vectorize_json(js, tag, data_structures, tagged=True, context_data=None):
    current_vectors = {}

    #for each key in the json provided, generate all the features required by the parameters
    for key in js.keys():
        # Lexical features are all generated from the data structure, which consists of dictionaries for each feature type
        # as the data structure only generates dictionaries for the features needed, bow_features only generates features
        # for the dictionaries available
        features = Features.bow_features(js[key]["words"], tagged_words=js[key]["tagged_words"], data_structures=data_structures)

        # TIME - one hot vector of length BUCKETES_BEST
        # RTs - len 2 vector, first element is initial 'rt', second is other 'rt'
        # WEB - binary feature of whether tweet contains a URL
        # W2V - len 200 vector, sum of word embeddings for the tweet. TODO - Allow average instead of sum
        # NER - not yet implemented, len of possible entities vector, with +1 for each found
        # VIEWEG - not yet implemented, Vieweg's classifier predictions for [situational_awareness, personal, formal, subjective]
        # CONTEXT - Key terms for tweets spanning -context_size to +context_size

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
                active_key = Tools.find_key(key, context_data, context=i)
                if active_key:
                    sent_words = context_data[active_key]["words"]
                else:
                    #tweets near the beginning and end will get all zeros
                    sent_words = []
                #same lexical features
                features += Features.bow_features(sent_words,data_structures=data_structures)

        #Pick the correct tag for the training data! Works slightly different for "None" and other tags
        if tag != "none":
            if tag in js[key]["annotations"] or tag in [a.split("-")[0].lower() for a in js[key]["annotations"]]:
                features.append(1)
            else:
                features.append(0)
        else:
            if "none" not in js[key]["annotations"]:
                features.append(1)
            else:
                features.append(0)
            
        current_vectors[key] = features
    return current_vectors


#Main method as suggested by van Rossum, simplified                                                                                            
def main(argv=None):
    def print_help():
        print ("CHIME-ML - Tweet Classification for Natural Disasters")
        print ("2016")
        print ("")
        print ("A filename can be specified as an optional parameter. Otherwise the default file will be used")
        print ("")
        print ("Optional Parameters")
        print ("-p, --param : specifies the parameter to use for the ML algorithm")
        print ("-t, --tag   : specifies which tag to classify")
        print ("")
        print ("Questions : kevin.stowe@colorado.edu")
        print ("")

    if argv is None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:], "hp:t:", ["help", "param:", "tag:"])
    except:
        print ("Error in args : " + str(argv[1:]))
        return 2

    param = None
    tag = "none"
    for o in opts:
        if o[0] == "-h" or o[0] == "--help":
            print_help()
            sys.exit(0)
        if o[0] == "-p" or o[0] == "--param":
            param = float(o[1])
        if o[0] == "-t" or o[0] == "--tag":
            tag = o[1].lower()
            if "-" in o[1]:
                tag = tag.split("-")[0]
            elif "_" in o[1]:
                tag = tag.split("_")[0]

    if len(args) == 0:
        print (run_cv(param=param,tag=tag))
    if len(args) == 1:
        print (run_cv(data=args[0],param=param,tag=tag))

    return 0

if __name__ == "__main__":
    sys.exit(main())
