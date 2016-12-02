# chime-ml
Code for Stowe et al 2016

The code is now functional, provided all the necessary packages are installed.

<h3>CURRENTLY REQUIRES</h3>
<h4>Packages</h4>
<a href="https://www.python.org/downloads/">Python</a>, tested on 3.4<br>
<a href="https://radimrehurek.com/gensim/">GenSim</a>, for the Word2Vec model<br>
<a href="http://www.nltk.org/install.html">NLTK</a>, for text normalization<br>
<a href="http://scikit-learn.org/stable/install.html">SciKit-Learn</a>, for machine learning algorithms (SVM/Naive Bayes/LogReg)<br>
<a href="http://www.numpy.org/">Numpy</a>, for support. SciKit-Learn or NLTK installations should include numpy/scipy.<br>

<h4>Extras</h4>
<h5>Word2Vec model</h5>
Our Twitter-specific word embedding model is not available on GitHub - its just too large. This will be fixed soon! For now, you'll have the change the <code>model</code> attribute of the Features class to point to a valid gensim Word2Vec model.
<br>
<h5>Tweet texts</h5>
We are not able to directly provide Tweet texts - as users may make tweets private or delete them, we instead only provide tweet ids. This allows users to collect available tweets from Twitter without unnecessarily exposing user data.
<br>
Because of this, the data provided contains an empty 'text' field. This field should be filled with Tweet texts collected from Twitter and tagged with the <a href="https://github.com/aritter/twitter_nlp">Twitter-NLP tagger</a>, with both --pos and --chunk flags. <br>

The 'text' field for each tweet should look like this:<br>

'text':'Just/O/RB/B-ADVP posted/O/VBD/B-VP a/O/DT/B-NP photo/O/NN/I-NP @/O/IN/B-PP Eight/O/NNP/B-NP Mile/B-geo-loc/NNP/I-NP River/I-geo-loc/NNP/I-NP http://t.co/1nkkwsIZ/O/URL/I-NP'<br>

This field is then parsed into POS and NE tags. This is done by splitting on "/", with the [-2] element being the POS tag and the [-3] element being the NE tag. [0:-3] are joined into the lexical item, and normalized.<br>

As an alternative, one could format their tweets with dummy values for POS and NER <br>
'text':'Just/0/0/0 posted/0/0/0 a/0/0/0 photo/0/0/0 @/0/0/0 Eight/0/0/0 Mile/0/0/0 River/0/0/0 http://t.co/1nkkwsIZ/0/0/0'<br>

This allows the featurizer to parse the next, without POS or NER features.

