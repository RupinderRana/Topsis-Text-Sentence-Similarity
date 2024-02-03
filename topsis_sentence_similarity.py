import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model_names = [
    "facebook/bart-base",
    "allenai/longformer-base-4096",
    "google/electra-small-discriminator",
    "microsoft/mpnet-base",
    "squeezebert/squeezebert-uncased",
    "deepset/sentence_bert",
    "vinai/phobert-base",
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased",
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
]

parameters = ["cosine_similarity", "euclidean_distance", "manhattan_distance", "minkowski_distance", "correlation_coefficient"]

data = []

paragraph1 = """
Natural language processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and contextually relevant.
"""

paragraph2 = """
Machine learning is a subset of artificial intelligence that involves the development of algorithms and statistical models that enable computers to perform specific tasks without explicit programming. In the context of natural language processing, machine learning algorithms are often used to analyze and understand the structure and meaning of human language.
"""

for model_name in model_names:
    model = SentenceTransformer(model_name)

    cosine_sim = cosine_similarity(model.encode([paragraph1]), model.encode([paragraph2]))[0][0]
    euclidean_dist = np.linalg.norm(model.encode([paragraph1]) - model.encode([paragraph2]))
    manhattan_dist = np.abs(model.encode([paragraph1]) - model.encode([paragraph2])).sum()
    minkowski_dist = np.power(np.power(np.abs(model.encode([paragraph1]) - model.encode([paragraph2])), 3).sum(), 1/3)
    jaccard_sim = len(set(paragraph1.split()) & set(paragraph2.split())) / len(set(paragraph1.split()) | set(paragraph2.split()))
    correlation_coeff = np.corrcoef(model.encode([paragraph1])[0], model.encode([paragraph2])[0])[0, 1]

    parameter_values = [cosine_sim, euclidean_dist, manhattan_dist, minkowski_dist, correlation_coeff]

    data.append([model_name] + parameter_values)

columns = ["Model"] + parameters
df = pd.DataFrame(data, columns=columns)

df_normalized = df.copy()
for param in parameters:
    df_normalized[param] = (df[param] - df[param].min()) / (df[param].max() - df[param].min())

criteria_weights = [1] * len(parameters)

weighted_normalized_matrix = df_normalized.iloc[:, 1:] * criteria_weights

positive_ideal_solution = weighted_normalized_matrix.max(axis=0)
negative_ideal_solution = weighted_normalized_matrix.min(axis=0)

distance_positive_ideal = np.linalg.norm(weighted_normalized_matrix - positive_ideal_solution, axis=1)
distance_negative_ideal = np.linalg.norm(weighted_normalized_matrix - negative_ideal_solution, axis=1)

df_normalized["TOPSIS_Score"] = df_normalized.apply(lambda row: np.sqrt(np.sum((row - positive_ideal_solution) ** 2)), axis=1)

df_ranked = df_normalized.sort_values(by="TOPSIS_Score", ascending=False).reset_index(drop=True)

df_ranked.to_csv("topsis_results.csv", index=False)
