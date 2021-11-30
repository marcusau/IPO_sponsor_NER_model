# Introduction of Repository

The repository is a reproduction and fork of https://github.com/Gxzzz/BiLSTM-CRF , while the data input and data type are different from original purpose.

This is a Pytorch implementation of BiLSTM-CRF for Named Entity Recognition, which is described in Bidirectional LSTM-CRF Models for Sequence Tagging.

The corpus is targeting English version of IPO prospectus of HKEX ,thus, no Chinese NLP tool is used in this repository



# Main dependencies
- Pytorch 1.8 (https://pytorch.org/get-started/previous-versions/)
- installation : pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html (cpu version)
- The model was trainied on google colab, where pytorch has been pre-installed in the platform.


# Usage of this repository


# key components of repository
- raw text data and labeled 
- corpus creation and textual data conversion to IOBES format
- model training
- model testing


# Corpus creation
- https://github.com/Wadaboa/ner-annotator


# model training


# model testing and deployment
