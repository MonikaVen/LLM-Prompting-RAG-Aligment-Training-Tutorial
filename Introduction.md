# Introduction

LLMs have been transforming the way we process and manipulate language over the past few years.
They are the revolution of Natural Language Processing. These deep neural networks are cabaple of more complex tasks, such as summarization, parsing complex data and answering complex questions.

The power of LLMs comes from both the vast amount of data they were trained on and the transformer architecture [1]. They often can have from millions to hundreds of billions of parameters that encode the relations and positions between parts of words (oftenly called tokens) and are optimized for the next word prediction in a sequence.
LLMs are a type of Artificial Intelligence that is called Generative AI.

# NLP

# Recurrent Neural Networks

## LSTM

# Intro to Generative AI

# Intro to Transformer Architecture

## Encoder and Decoder

Encoder module encodes the input text as a series of vector representations. Decoder decodes these representations into human-readable sequences of words. Both the encoder and decoder are made of layers of self-attention mechanism.

## BERT - Biderectional Encoder Representations from Transformers

Masked word prediction. Primary used for text classification tasks.

## GPT - Generative Pretrained Transformer

Learns to generate one word at a time. Next word prediction is a form of self-supervised learning. Machine translation, text summarization, code generation, creative writing, zero-shot, few-shot generation and etc.

### GPT Architecture

Generates text by predicting one word at a time. 
Auto-regressive model - incorporates previous outputs as inputs for future predictions.
(Utilizes only the Decoder part from the original transformer architecture.)

GPT-3, introduced in 2020, has 96 transformer layers and 175 billion parameters.

# Building an LLM

## Data Preparation

## Attention Mechanism

## LLM Architecture

## Pretraining

## Training Loop

## Model Evaluation

## Loading Pretrained Model Weights

## Finetuning a Classifier

## Finetuning a Personal Assistant

# Prompting an LLM

# Training an LLM

## Pretraining an LLM

Creating a base foundation model. This model is trained on a vast amounts of unlabeled data and has text completion and limited few-shot capabilities (learning from a few examples).

## Finetuning an LLM

Next phase involves finetuning

### Instruction Finetuning

### Classification

# Retrieval Augmented Generation with LLM

# Libraries that Were Used

PyTorch
LLamaIndex

# Common Open-source LLM Weights

# Publicly Available Datasets

Dolma

# Resources for Further Study
## Books

## Papers

(1) Attention Is All You Need (2017) by Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin, https://arxiv.org/abs/1706.0376211

(2) Improving Language Understanding by Generative Pre-Training (2018) by Radford, Narasimhan, Salimans and Sutskever, https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
