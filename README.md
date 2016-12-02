# chime-ml
Code used in production of <a href="http://aclweb.org/anthology/W/W16/W16-6201.pdf">Stowe et al 2016</a>. There've been significant bug fixes and processing improvements since the paper, but the dataset, methods, and features are the same.


<h3>USAGE</h3>
python CHIME-ML.py (-p or --param FLOAT) (-t STRING) (datafile)

The -p argument can be used to specify the main parameter the ML algorithm uses - it will change depending on the algorithm.

The -t argument can be used to specify which tag to classify - this can be any of the high level tags
<ul>
<li>Sentiment</li>
<li>Information</li>
<li>Reporting</li>
<li>Movement</li>
<li>Preparation</li>
<li>Actions</li>
</ul>

Finally, a dataset file may be passed - if not, it will use the default file set in CHIME-ML.py.

<h3>Code Overview</h3>
Takes a .json object (defaulting to the provided 'data/part1/cleaned.json') of tweets. The object contains is keyed by tweet_id.

<code>
tweet_id:{'text':''*, 'geo_coords':'[lat, long]' or '[]', 'user':'user name', 'date':'MM-DD-YYYY HH:MM:SS', 'annotations':[list of possibles anns, or one element "None"], 'previous':'tweet_id of previous tweet in user stream', 'next':'tweet_id of next tweet in user stream'}
</code>

The json is loaded, featurized according the parameters of the CHIME-ML.py script using the Features.py module, and then run through 5-fold CV using the Learn.py module. The particular ML algorithm can be specified as ALGORITHM parameter (either 'SVM','NB', or 'LR'). It returns a dictionary of tweet_id:binary prediction, as well as F1, precision, and recall.

<h3>CURRENTLY REQUIRES</h3>
<h4>Packages</h4>
<a href="https://www.python.org/downloads/">Python</a>, tested on 3.4<br>
<a href="https://radimrehurek.com/gensim/">GenSim</a>, for the Word2Vec model<br>
<a href="http://www.nltk.org/install.html">NLTK</a>, for text normalization<br>
<a href="http://scikit-learn.org/stable/install.html">SciKit-Learn</a>, for machine learning algorithms (SVM/Naive Bayes/LogReg)<br>
<a href="http://www.numpy.org/">Numpy</a>, for support. SciKit-Learn or NLTK installations should include numpy/scipy.<br>

<h3>Extras</h3>
<h4>*Tweet texts</h4>
We are not able to directly provide Tweet texts as users may make tweets private or delete them. Instead, we provide all of our metadata, along with tweet ids. This allows collection of available tweets via Twitter without unnecessarily exposing user data.
<br>
Because of this, the data provided (data/part1/cleaned.json) contains an empty 'text' field. This field should be filled with Tweet texts collected from Twitter and tagged with the <a href="https://github.com/aritter/twitter_nlp">Twitter-NLP tagger</a>, with both --pos and --chunk flags. <br>

The 'text' field for each tweet should look like this:<br>

'text':'Just/O/RB/B-ADVP posted/O/VBD/B-VP a/O/DT/B-NP photo/O/NN/I-NP @/O/IN/B-PP Eight/O/NNP/B-NP Mile/B-geo-loc/NNP/I-NP River/I-geo-loc/NNP/I-NP http://t.co/1nkkwsIZ/O/URL/I-NP'<br>

This field is then parsed into POS and NE tags. This is done by splitting on "/", with the [-2] element being the POS tag and the [-3] element being the NE tag. [0:-3] are joined into the lexical item, and normalized.<br>

As an alternative, one could format their tweets with dummy values for POS and NER <br>
'text':'Just/0/0/0 posted/0/0/0 a/0/0/0 photo/0/0/0 @/0/0/0 Eight/0/0/0 Mile/0/0/0 River/0/0/0 http://t.co/1nkkwsIZ/0/0/0'<br>

This allows the featurizer to parse the next, without POS or NER features.

<h4>Word Embeddings</h4>
The word embedding model is provided by <a href="https://git-lfs.github.com/">Git's Large File Storage</a>


<h5>Please send any and all questions to:<br></h5>
kevin.stowe@colorado.edu
