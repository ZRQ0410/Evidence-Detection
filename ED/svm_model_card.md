---
{}
---
language: en
license: cc-by-4.0
tags:
- pairwise-sequence-classification
repo: https://drive.google.com/file/d/17_HhZsecvZyuXlb1kCIQlgG5hDjMHMib/view?usp=drive_link

---

# Model Card for p37245yz-r67061rz-ED_SVM(Tf-idf)

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether the evidence is relevant to the claim.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a support vector machine model (SVM) with Term Frequency-inverse Document Frequency (TF-IDF) that was fine-tuned
      on 23K pairs of texts.

- **Developed by:** Yaqi Zhao and Ruoqing Zheng
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Support Vector Machine
- **Finetuned from model [optional]:** Support Vector Machine

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://scikit-learn.org/stable/modules/svm.html
- **Paper or documentation:** https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=708428

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

This model was trained on all of 23K claim-evidence pairs provided as the training dataset.
    Text preprocessing was applied to the dataset, including converting all data to lower case, removing punctuation, tokenizing and lemmatizing.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - kernel: rbf
      - C: 10
      - gamma: 1

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 8 min
      - model size: 7.16MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

All data in the development set were tested, including around 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - Precision
      - Recall
      - F1-score
      - ROC AUC

### Results

The model obtained an accuracy of 0.834, an Precision of around 0.801,    an Recall of 0.756, an F1-score of 0.774, an ROC AUC of 0.756.

## Technical Specifications

### Hardware


      - RAM: at least 2 GB
      - Storage: at least 30 GB
      - CPU

### Software


      - Scikit-learn 1.4.2
      - NLTK 

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

 OOV(Out of Vocabulary) might occur due to the limitation of Tf-idf.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The model selection were determined by experimentation. The hyperparameters were selected by Grid Search with different values.
