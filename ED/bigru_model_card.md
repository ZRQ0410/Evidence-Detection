---
{}
---
language: en
license: cc-by-4.0
tags:
- pairwise-sequence-classification
repo: https://colab.research.google.com/drive/1y4Pai6R6G_wMSjiqvp2MXzoT0E5rMXMP?usp=sharing

---

# Model Card for p37245yz-r67061rz-ED_BiGRU(GloVe)

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether the evidence is relevant to the claim.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a Bidirectional Gated Recurrent Units model (BiGRU) with Global Vectors (GloVe) embedding that was fine-tuned
      on 23K pairs of texts.

- **Developed by:** Yaqi Zhao and Ruoqing Zheng
- **Language(s):** English
- **Model type:** Recurrent Neural Network
- **Model architecture:** Bidirectional GRU
- **Finetuned from model [optional]:** GRU

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://keras.io/api/models/
- **Paper or documentation:** https://doi.org/10.48550/arXiv.2109.05346

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

This model is trained with 23K claim-evidence pairs text data which is tokenized to a sequence     by keras.preprocessing.text.Tokenizer and padded by keras.preprocessing.sequence.pad_sequences.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - optimizer: 'adam'
      - loss: 'binary_crossentropy'
      - dropout: 0.7
      - gru_units: 128
      - dense: 128
      - max_len: 307

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 60 min
      - duration per training epoch: 12 minutes
      - model size: 190MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

The development set provided, amounting to 6K pairs, which is preprocessed by the same steps.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - Precision
      - Recall
      - F1-score
      - ROC AUC

### Results

The model obtained an accuracy of 0.831, a precision of 0.798, a recall of 0.750, a F1-score of 0.768, and a ROC AUC of 0.750.     This accuracy of the result is around 10% higher than the basic GRU model.

## Technical Specifications

### Hardware


      - RAM: 8 GB
      - Storage: 20 GB
      - CPU

### Software


      - Keras
      - Tensorflow

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any word that is not included in GloVe embedding matrix will have no pre-trained embedding value.     Any input (single sequence) longer than the max length of training data 307 words will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The structure (eg. the layers) of the model and the hyperparameters were determined by experimentation
      with different values.
