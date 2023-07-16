# Genomic MSA Transformer

This repository contains the code for the **Genomic MSA Transformer** project. The project uses DNA sequences in a multiple sequence alignment transformer. These are trained in an unsupervised fashion. A classifier then uses the embeddings from the transformer to classify operons within genomes of various organisms.

## Table of Contents

- [Paper](#paper)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Paper

### For more details on the project, please refer to our paper: [Learning Genome Architecture Using MSA Transformers](https://github.com/EmaadKhwaja/Genomic_MSA_Transformer/blob/main/paper/CS294_Final_Paper.pdf)

![alt text](paper/images/Model%20Diagram.png)

## Installation

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To train the model, run:

```bash
python train.py
```

To test the model, run:

```bash
python test.py
```

## Results

The model achieved an accuracy of 90% on the test set.
```

I hope this helps! Let me know if you have any other questions.
