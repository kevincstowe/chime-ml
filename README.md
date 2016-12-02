# chime-ml
Code for Stowe et al 2016

The code is now functional, provided all the necessary packages are installed.
CURRENTLY REQUIRES
python 3
<a href="https://radimrehurek.com/gensim/">GenSim, for the Word2Vec model</a>
NLTK
numpy
sklearn

<b>HOWEVER</b>
We are not able to directly provide Tweet texts - as users may make tweets private or delete them, we instead only provide tweet ids. This allows users to collect available tweets from Twitter without unnecessarily exposing user data.

Because of this, the data provided contains an empty 'text' field. This field should be filled with Tweet texts collected from Twitter and tagged with the <a href="https://github.com/aritter/twitter_nlp">Twitter-NLP tagger</a>, with both --pos and --chunk flags. 
The 'text' field for each tweet should look like this:
'text':'Just/O/RB/B-ADVP posted/O/VBD/B-VP a/O/DT/B-NP photo/O/NN/I-NP @/O/IN/B-PP Eight/O/NNP/B-NP Mile/B-geo-loc/NNP/I-NP River/I-geo-loc/NNP/I-NP http://t.co/1nkkwsIZ/O/URL/I-NP'

This field is then parsed into POS and NE tags. This is done by splitting on "/", with the [-2] element being the POS tag and the [-3] element being the NE tag. [0:-3] are joined into the lexical item, and normalized.

As an alternative, one could format their tweets with dummy values for POS and NER 
'text':'Just/0/0/0 posted/0/0/0 a/0/0/0 photo/0/0/0 @/0/0/0 Eight/0/0/0 Mile/0/0/0 River/0/0/0 http://t.co/1nkkwsIZ/0/0/0'

This allows the featurizer to parse the next, without POS or NER features.

