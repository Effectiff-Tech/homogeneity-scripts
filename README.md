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


## Installation
Clone the project and install the requirements:
```
git clone https://github.com/Effectiff-Tech/homogeneity-scripts.git

cd homogeneity-scripts && pip install -r requirements.txt
```
## Usage
```bash
python clustering.py -i bigdata.docx
```
Note: Run `python clustering.py -h` to see full set of options.

## License
Licensed under the [MIT](LICENSE) License.
