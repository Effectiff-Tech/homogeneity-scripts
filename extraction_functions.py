import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
import string
from blingfire import *
from topia.termextract import extract, tag
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
import os
from pyate import combo_basic
import re

# Для извлечения отдельных слов исключаем 1000 самых частотных слов английского языка через стоп-слова
with open("en1000.txt", encoding="utf-8-sig") as file:
    new_stops = file.read().splitlines()

contractions = ["'ll", "'d", "'s", "'re", "'ve", "'m", "n't", "nt", "’s"]

stopset_iw = set(stopwords.words('english') + list(string.punctuation) + new_stops + contractions)

stopset_nltk_punkt = set(stopwords.words('english') + list(string.punctuation) + contractions)
stop_characters = "«»\'!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~’“”‘01234567890□"
filter_stops = lambda w: (w in stopset_nltk_punkt) or (all(c in stop_characters for c in w))

tagger = tag.Tagger()
tagger.initialize()
extractor = extract.TermExtractor(tagger)

vectorizer = KeyphraseCountVectorizer()

kw_model = KeyBERT()

def output_with_fr(bcf, output):
    output_lr = []
    for i in output:
        for u in list(bcf.ngram_fd.items()):
            if i == u[0]:
                output_lr.append((i, u[1]))
    return output_lr

def individual_words(text):
    Btokenized = text_to_words(text)
    words = nltk.word_tokenize(Btokenized)
    myTokenFD = nltk.FreqDist([word for word in words if (word not in stopset_iw) and not (all(c in stop_characters for c in word))])
    terms = set()
    total_count = 0
    for term, freq in myTokenFD.most_common(30):
        terms.add(term)
        total_count += freq  
    return terms, total_count

def coll_likelihood_ratio(text):
    Btokenized = text_to_words(text)
    
    words = nltk.word_tokenize(Btokenized)

    bcf = BigramCollocationFinder.from_words(words)

    bcf.apply_word_filter(filter_stops)

    output = bcf.nbest(BigramAssocMeasures.likelihood_ratio, 30)

    output_lr = output_with_fr(bcf, output)
    terms = set()
    total_count = 0
    for term, freq in output_lr:
        terms.add(" ".join(term))
        total_count += freq
        
    return terms, total_count

def coll_pmi(text):
    Btokenized = text_to_words(text)
    
    words = nltk.word_tokenize(Btokenized)

    bcf = BigramCollocationFinder.from_words(words)

    bcf.apply_word_filter(filter_stops)

    output = bcf.nbest(BigramAssocMeasures.pmi, 30)

    output_lr = output_with_fr(bcf, output)
    terms = set()
    total_count = 0
    for term, freq in output_lr:
        terms.add(" ".join(term))
        total_count += freq
        
    return terms, total_count

def terms_topia(text):
    Btokenized = text_to_words(text)
    terms = set()
    total_count = 0
    raw_terms = extractor(Btokenized)
    # Теперь проверяем термины на стоп-слова
    filtered_terms = [raw_term for raw_term in raw_terms if not any((i in stopset_nltk_punkt) or all(c in stop_characters for c in i) for i in raw_term[0].split())]
    for ex in sorted(filtered_terms, key=lambda x: (x[1], x[2]), reverse=True)[:30]:
        terms.add(ex[0])
        total_count += ex[1]    
        
    return terms, total_count

def vector_keybert(text):
    Btokenized = text_to_words(text)
    vectorizer.fit([Btokenized])
    keyphrases = vectorizer.get_feature_names_out()
    document_keyphrase_matrix = vectorizer.transform([Btokenized]).toarray()
    terms_with_freq = []
    for a, b in zip(keyphrases, document_keyphrase_matrix[0]):
        terms_with_freq.append((a, b))
    terms = set()
    total_count = 0
    for keyphrase, score in kw_model.extract_keywords(Btokenized, top_n=30, vectorizer=KeyphraseCountVectorizer()):
        for np in terms_with_freq:
            if keyphrase == np[0]:
                terms.add(keyphrase)
                total_count += np[1]            
    return terms, total_count

def clean_bigrams(text):
    Btokenized = text_to_words(text)
    
    tokens = nltk.word_tokenize(Btokenized)

    bigrams = nltk.bigrams(tokens)

    frequence = nltk.FreqDist(bigrams)

    filtered_freq = dict((" ".join(phrase), freq) for phrase, freq in frequence.most_common() if (phrase[0] not in stopset_nltk_punkt) and not (all(c in stop_characters for c in phrase[0])) and (phrase[1] not in stopset_nltk_punkt) and not (all(c in stop_characters for c in phrase[1])))
    terms = set()
    total_count = 0
    for term, freq in filtered_freq.items():
        terms.add(term)
        if freq > 1:
            total_count += freq
    return terms, total_count

def pyate_combo(text):
    Btokenized = text_to_words(text)
    
    raw_terms = combo_basic(Btokenized, have_single_word=True).sort_values(ascending=False).index.values
    
    terms = set()
    total_count = 0
    # Теперь проверяем термины на стоп-слова
    filtered_terms = [raw_term for raw_term in raw_terms if not any((i in stopset_nltk_punkt) or all(c in stop_characters for c in i) for i in raw_term.split())]

    for combo in filtered_terms[:30]:
        terms.add(combo)
        total_count += len(re.findall(fr'\b{combo}\b', Btokenized))  
    
    return terms, total_count