# Problem to solve: 
In this assessment my goal is to take written clinical notes about patients and determine if they have:
1. Cancer (Yes / No)
2. Diabetes (Yes / No)

Framing this as a Data Science / Machine Learning problem, attempt to build a model that will take as input a string that gives the medical details of a patient, and outputs 2 scores, one score to represent the probability the patient has cancer (given the clinical notes), and one score to represent the probability that the patient has diabetes. While the training labels are given as binary values (0 or 1), I will keep the outputs as probabilities that can take values between 0-1. I believe that in this format the outputs will also show a level of uncertainty in the predictions, since a prediction of .999 and .701 may both map to a "1", the .999 may be considered more strongly.


# Data
The text is taken from the [Asclepius Synthetic
Clinical Notes](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes) dataset.

The data given is a table containing the columns: 
* patient_identifier: id
* text: clinical notes that will be used as input to the model
* has_cancer: binary label showing if patient has cancer or not; used as training label
* has_diabetes: binary label showing if patient has diabetes or not; used as training label


However, it should be noted that in this data set, only 50 of the 1800 total data points contain labels, with the remaining 1750 data points missing labels. 
Therefore, I take 35 of the labelled data points to be used for fine-tuning and keep 15 of the labelled data points to be used for evaluation.


Furthermore, there are 200 datapoints marked for a test set, which I remove from training and put into a separate data set.

# Methodology
For this analysis I begin with the pretrained Huggingface model meta-llama/Meta-Llama-3-8B as the base LLM and tokenizer. Being a relatively small, state-of-the-art open source model, I believe that it will generate good representations of the input text.

I then apply LoRA adapters to the base model, which allows me to fine-tune the model on our data by updating only a small percentage of the total model weights.

I then tokenize the text, including padding and truncation to sequence lengths of 1024 since this covers almost all the token lengths of the text (see sequence_lengths.png). Once the text is tokenized I can fine-tune the LLM. 

After fine-tuning the LLM, I add a classification head to it, which takes the final text vector representations and outputs probability scores for cancer and diabetes. I then train the classification head on the labelled datapoints. After training the classification head, I predict probabilties for cancer and diabetes for the unlabelled training data. I set a confidence threshold for the predictions, which takes a value between [0, .5] and looks at the values:
* predicted_probability_cancer
* 1 - predicted_probability_cancer (probability of no cancer)
* predicted_probability_diabetes
* 1 - predicted_probability_diabetes (probability of no diabetes)

If any of these values are less than the confidence threshold, the data point is considered "high-confidence" and added to the training dataset. The "low-confidence" data points have their predictions removed. The idea of the confidence is that if the probability is very close to 0, the model is confident that cancer / diabetes is not present. If the probability is very close to 1, the model is very confident that cancer / diabetes is present. The closer the probability is to .5, the less confident the model is. With the new training set and new unlabelled dataset, I repeat this training process until either all the unlabelled data are confidently labelled or a set number of iterations have passed.

# Error Metrics
Below are metrics calculated for the trained model performance on the 15-datapoint evaluation set. Something that stands out to me is the relatively high recall scores, which show that the model is good at recognizing positive samples (i.e. the presence of cancer / diabetes). For a medical use-case this is good behavior since we would rather have more False Positives than False Negatives; we'd rather assume someone has cancer then disprove it, instead of assume they don't have cancer only to find out that they do.


However, it seems that the model may be over-zealously predicting positive cases, which is represented by the low accuracy and precision scores. Particularly for Diabetes, it would be easy for the model to have high accuraccy by simply always predicting "no diabetes"; due to the data imbalance this would result in 90% accuracy. 

| Condition     | Accuracy      | Precision     |   Recall      |   F1 - Score  | MSE           | AUROC         | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Cancer        | 0.4           | 0.3846        | 0.8333        | 0.5263        | 0.2978        | 0.4352        |
| Diabetes      | 0.1333        | 0.1333        | 1.0           | 0.2353        | 0.5163        | 0.2692        |

# Future Steps
## Continued Pretraining:
In this assignment, I utilized the unlabelled data by predicting pseudo-labels and then using them as training data in a self-learning method. I believe that a better utilization of the unlabelled data would be to perform additional pretraining on the base LLM on masked inputs of the data, so that the model will have better familiarity with medical-specific text. It may not be necessary to update all weights in the pretraining, a parameter we could choose is the subset of weights to update during this additional pretraining.

## Data Imbalance: 
Even among the labelled data, there is a large amount of imbalance. For the cancer labels, only 20 / 50 data points are in the positive class, while in the diabetes labels only 5 / 50 are in the positive class! The data imbalance will make it difficult to train the model to recognize the minority classes. In the future I would like to implement procedures to mitigate this imbalance. Some possible methods would be:
* Oversampling / Undersampling
* SMOTE; unsure how this would work with text inputs, perhaps could be used on tokenized data
* Class weights

## Other Architectures / LLM's
I would like to repeat these experiments on different LLM's to compare performance

## Iterative Confidence updating:
Currently I add a new pseudo-labelled datapoint to the labelled training dataset if either the cancer prediction or diabetes prediction is confident enough. While one of the predictions may be confident, the other one may not be. In the case that only one prediction is confident, it may be useful to be able to add only one label to the training set, and add the other when it is confident enough.

