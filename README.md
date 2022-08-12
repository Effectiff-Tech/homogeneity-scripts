![Logo](https://raw.githubusercontent.com/Effectiff-Tech/homogeneity-scripts/main/img/logo.png)
# homogeneity-scripts
Code in connection with Effectiff's research on evaluating homogeneity among a set of documents.
- [Introduction](#introduction)
- [Method](#method)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction
The homogeneity of two or more texts is established on the following principles:
- The same domain and genre.
- Similar readability scores based on selected metrics.
- Closeness in the density (number of occurrences) of specialized terminology.
- No (very few) overlapping specialized terms.

## Method

Initially, it makes sense to take texts of approximately the same structure and style of presentation. To maintain style uniformity, it is convenient to take the texts of one author.

It is important that the initial corpus is large enough to allow splitting into a multitude of pieces from which the most homogeneous pieces can be selected as compared to other pieces.

The texts are divided into paragraphs. Each piece is formed when the number of words in the selected consecutive paragraphs became more than the set value. Thus, comparable pieces are obtained, while retaining the division into paragraphs.

The selected pieces then undergo tokenization and segmentation (division into sentences) using the [BlingFire](https://github.com/microsoft/BlingFire) library.

Then the readability metrics and general textual statistical metrics are calculated for each piece. We use the following metrics as features for further clustering:

- [Flesch Reading Ease](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
- [LIX](https://en.wikipedia.org/wiki/Lix_(readability_test))
- [Dale-Chall Index](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula)
- Characters per word
- [Type token ratio](https://en.wikipedia.org/wiki/Lexical_density)

The first three metrics are based on the **average number of words in a sentence**, and also include the **average number of syllables in a word** (_Flesch Reading Ease_), the proportion of long words (_LIX_), or the **proportion of "difficult" words** (_Dale-Chall Index_). In addition, metrics related to the **number of characters in a word** (_Characters per word_) and the **proportion of different words** (_Type token ratio_) are also used. Thus, a wide range of sentence-level and word-level parameters are involved to assess homogeneity.

Metric values are then normalized using the [min-max or z-score method](https://en.wikipedia.org/wiki/Feature_scaling) and grouped into clusters using the [DBSCAN algorithm](https://en.wikipedia.org/wiki/DBSCAN) (allowing selection of clusters with closely spaced feature values, but not necessarily assigning each piece to a cluster). To determine the optimal eps distance, the [kneedle algorithm](https://github.com/arvkevi/kneed) is used. In fact, eps values less than the optimal distance are recommended to be taken (to distinguish small clusters of particularly close texts). The distance between the points is calculated as a Euclidean metric, and the minimum number of pieces in the cluster is also to be set to a desired value. 

The obtained clusters of homogeneous pieces are then checked for properties related to terminology based on the two stated principles: 
1. homogeneous texts should not have a (significant) number of shared terms
2. homogeneous text should have approximately the same density of terminology. 

Here we are faced with the problem of the lack of a single reliable and universally accepted method of term extraction. Supposedly, the _TF-IDF_ method could be used as the “gold standard” for extracting terminology. But in our case, applying that method was rather tenuous due to the fact that it is not clear what to take as a corpus of documents for calculating _IDF_. If we take the documents of one cluster, it is easy to imagine the situation that such cluster includes pieces closely related in terminology, then even characteristic terms specific to all or most of such pieces will get a low _IDF_ value and will not be defined as the best term candidates, which would be incorrect. We could use all the pieces of all the documents used in the clustering, or even the available _IDF_ values for large English corpora as the basis for _IDF_ calculation, but this would be quite arbitrary.

Therefore, other approaches are applied, which are based on the identification of:
1. just **individual frequent words** (excluding the 1000 most frequent words of the English language based on the data from www.wordfrequency.info)
2. collocations (using both [**likelihood ratio**](https://en.wikipedia.org/wiki/Likelihood-ratio_test) and [**pointwise mutual information**](https://en.wikipedia.org/wiki/Pointwise_mutual_information) metrics) 
3. terms based on POS properties ([**topia.termextract**](https://pypi.org/project/topia.termextract/), [**KeyphraseVectorizers**](https://github.com/TimSchopf/KeyphraseVectorizers) with [KeyBERT](https://github.com/MaartenGr/KeyBERT), [**cosmo_basic**](https://arxiv.org/abs/1611.07804) algorithm as implemented in [pyate](https://github.com/kevinlu1248/pyate))
4. **all suitable bigrams**

Thus, a total of 7 methods of term extraction are used.

The initially selected terms are also filtered by a list of stop words. A term candidate is excluded if at least one of its words is a stop word.

## Installation
Clone the project and install the requirements:
```
git clone https://github.com/Effectiff-Tech/homogeneity-scripts.git

cd homogeneity-scripts && pip install -r requirements.txt
```
## Usage
```bash
python clustering.py -i corpus/bigdata.docx
```
Note: Run `python clustering.py -h` to see full set of options.

## License
Licensed under the [MIT](LICENSE) License.
