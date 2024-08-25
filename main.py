from ollama import Client
from httpx import BasicAuth
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools

# Initialize Ollama Client
ollama = Client(host='https://ollama.doze.dev:443', auth=BasicAuth(username="admin", password="not_my_real_password"))

words = [
    "steaming",
    "poker",
    "air",
    "baking",
    "laundry",
    "ranking",
    "boiling",
    "station",
    "heating",
    "origami",
    "position",
    "fuming",
    "standing",
    "ventilation",
    "livid",
    "conditioning"
]

# Get embeddings for each word
embeddings = {
    word: ollama.embeddings(
        model='llama3.1',
        prompt=word
    )['embedding'] for word in words
}

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(list(embeddings.values()))

# Function to score a single cluster
def score_cluster(cluster):
    indices = [words.index(word) for word in cluster]
    cluster_matrix = similarity_matrix[indices][:, indices]
    return np.sum(cluster_matrix)

# Generate and score all possible groups of 4 words
group_scores = []
for group in itertools.combinations(words, 4):
    score = score_cluster(group)
    group_scores.append((group, score))

# Normalize scores
max_score = max(score for _, score in group_scores)
group_scores = [(group, score / max_score) for group, score in group_scores]

# Sort groups by their normalized score in descending order
group_scores.sort(key=lambda x: x[1], reverse=True)

# Function to check if a group overlaps with already selected words
def overlaps_with_selected(group, selected):
    return bool(set(group) & set(selected))

# Initialize the best score and combination
best_combination = None
best_score = -np.inf

# Find the best non-overlapping combination of 4 groups
for i, (group1, score1) in enumerate(group_scores):
    if score1 * 4 <= best_score:
        break  # Skip if it's impossible to beat the current best score
    
    selected_words = set(group1)
    remaining_score = score1

    for j, (group2, score2) in enumerate(group_scores[i+1:], start=i+1):
        if overlaps_with_selected(group2, selected_words):
            continue
        selected_words.update(group2)
        remaining_score += score2
        
        if remaining_score + score2 * 3 <= best_score:
            selected_words = set(group1)  # Reset selected words
            remaining_score = score1
            continue  # Skip if it's impossible to beat the current best score

        for k, (group3, score3) in enumerate(group_scores[j+1:], start=j+1):
            if overlaps_with_selected(group3, selected_words):
                continue
            selected_words.update(group3)
            remaining_score += score3
            
            if remaining_score + score3 * 2 <= best_score:
                selected_words = set(group1)  # Reset selected words
                remaining_score = score1 + score2
                continue  # Skip if it's impossible to beat the current best score

            for l, (group4, score4) in enumerate(group_scores[k+1:], start=k+1):
                if overlaps_with_selected(group4, selected_words):
                    continue
                total_score = remaining_score + score4
                
                # If this combination is the best found, record it and print
                if total_score > best_score:
                    best_combination = (group1, group2, group3, group4)
                    best_score = total_score
                    print(f"New best combination found with score {best_score:.4f}:")
                    for group in best_combination:
                        print(group)
                    print(i, j, k, l)
                    print(total_score)
                    print()

# Print the final best combination of groups
if best_combination:
    print("Best overall combination:")
    for group in best_combination:
        print(group)
else:
    print("No valid combination found.")
