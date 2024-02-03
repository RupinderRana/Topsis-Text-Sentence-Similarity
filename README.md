# Topsis-Text-Sentence-Similarity(102117128)

  <img width="851" alt="result_final" src="https://github.com/RupinderRana/Topsis-Text-Sentence-Similarity/assets/98392235/3837161f-71fb-4fef-a2f4-cd620119d1f3">


## Choosing models
### I selected following pre-Trained models from hugging face: 
  *  "facebook/bart-base",
  *  "allenai/longformer-base-4096",
  *  "google/electra-small-discriminator",
  *  "microsoft/mpnet-base",
  *  "squeezebert/squeezebert-uncased",
  *  "deepset/sentence_bert",
  *  "vinai/phobert-base",
  *  "bert-base-uncased",
  *  "roberta-base",
  *  "distilbert-base-uncased",
  *  "sentence-transformers/paraphrase-MiniLM-L6-v2"
    
## Choosing Parameters 
### I selected following parameters for text-sentence similarity:
 *  Cosine Similarity
 *  Euclidean Distance
 *  Manhattan Distance
 *  Minkowski Distance
 *  Correlation Coefficient

## Evaluating Model and Normalization
* Evaluate models and create a DataFrame with parameter values for each model.
* Normalize parameter values to a common scale.
* Assume equal weights for simplicity.
* Multiply normalized parameter values with criteria weights.
* Identify the maximum and minimum values for each criterion.
* Calculate the distance of each model from positive and negative ideal solutions.

## Ranking on the basis of TOPSIS score
* Calculate the TOPSIS scores for each model.
* Rank models based on their TOPSIS scores in descending order.
* Save the ranked results to a CSV file named "topsis_results.csv".


### I have selected "sentence-transformers/paraphrase-MiniLM-L6-v2" as best pre-trained model because of highest TOPSIS-Score (1.78760890209212)
  
