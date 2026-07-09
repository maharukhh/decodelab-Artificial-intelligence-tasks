"""
================================================================
 DecodeLabs - Artificial Intelligence Internship
 Project 3: AI Recommendation Logic
 Capstone: Tech Stack Recommender

 Goal: Map a user's raw skills / career interests to the most
       relevant job roles using Content-Based Filtering.

 Follows the IPO (Input -> Process -> Output) architecture and
 the 4-step ranking pipeline from the training slides:
   1. Ingestion  - capture user skills (min. 3 inputs)
   2. Scoring    - TF-IDF vectorization + Cosine Similarity
   3. Sorting    - rank job roles by similarity score (descending)
   4. Filtering  - return the Top-N (Top 3) matches
================================================================
"""

import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------
# STEP 0: Load the item dataset (job roles + their skill tags)
# ----------------------------------------------------------------
def load_dataset(path="raw_skills.csv"):
    df = pd.read_csv(path)

    def clean_skills(raw):
        # Split on commas to get individual skill tags, then join the
        # words *within* each tag using underscores (so "Machine Learning"
        # becomes one token "machine_learning"), and join tags with a
        # real space so TfidfVectorizer treats each tag as a separate token.
        tags = [t.strip().lower().replace(" ", "_").replace("/", "_") for t in raw.split(",")]
        return " ".join(tags)

    df["skills_clean"] = df["skills"].apply(clean_skills)
    return df


# ----------------------------------------------------------------
# STEP 1: Ingestion - capture user's raw skills (minimum 3)
# ----------------------------------------------------------------
def build_user_profile(user_skills):
    if len(user_skills) < 3:
        raise ValueError("Please provide at least 3 skills for accurate matching.")
    cleaned = [s.strip().lower().replace(" ", "_") for s in user_skills]
    return " ".join(cleaned)


# ----------------------------------------------------------------
# STEP 2, 3, 4: Scoring (TF-IDF + Cosine Similarity),
#               Sorting (descending), Filtering (Top-N)
# ----------------------------------------------------------------
def recommend_roles(df, user_skills, top_n=3):
    user_profile_text = build_user_profile(user_skills)

    # Combine item corpus + user profile into ONE shared vocabulary space
    # (the "Bridging the Language Barrier" step from the slides) so that
    # both are transformed using the exact same TF-IDF vocabulary.
    corpus = df["skills_clean"].tolist() + [user_profile_text]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    item_vectors = tfidf_matrix[:-1]      # all job roles
    user_vector = tfidf_matrix[-1]        # the last row is the user profile

    # Cosine Similarity: measures the ANGLE between vectors (0 to 1 here,
    # since TF-IDF values are non-negative) -> invariant to profile length.
    scores = cosine_similarity(user_vector, item_vectors).flatten()

    results = df.copy()
    results["match_score"] = scores
    results["match_percent"] = (results["match_score"] * 100).round(1)

    # SORT descending by score, then FILTER to Top-N
    ranked = results.sort_values(by="match_score", ascending=False)
    top_matches = ranked.head(top_n)

    return top_matches[["job_role", "skills", "match_percent"]]


# ----------------------------------------------------------------
# Cold-start guard: warn if score is 0 (no overlapping vocabulary)
# ----------------------------------------------------------------
def check_cold_start(top_matches):
    if top_matches["match_percent"].max() == 0:
        print(
            "\n[NOTICE] No meaningful overlap found between your skills and "
            "the dataset's vocabulary (Cold Start scenario). Try using more "
            "common/standard skill names, e.g. 'Python', 'SQL', 'AWS'."
        )


# ----------------------------------------------------------------
# MAIN - demo run with a few example user profiles
# ----------------------------------------------------------------
if __name__ == "__main__":
    df = load_dataset("raw_skills.csv")

    print("=" * 65)
    print("PROJECT 3: TECH STACK RECOMMENDER - Demo Run")
    print("=" * 65)
    print(f"Loaded {len(df)} job roles from raw_skills.csv\n")

    # ---- Example 1: matches the slide's own example -------------
    example_1 = ["Python", "Cloud", "Automation"]
    print(f"USER INPUT #1: {example_1}")
    result_1 = recommend_roles(df, example_1, top_n=3)
    check_cold_start(result_1)
    print(result_1.to_string(index=False))
    print("-" * 65)

    # ---- Example 2: a frontend-leaning profile -------------------
    example_2 = ["JavaScript", "React", "HTML"]
    print(f"\nUSER INPUT #2: {example_2}")
    result_2 = recommend_roles(df, example_2, top_n=3)
    check_cold_start(result_2)
    print(result_2.to_string(index=False))
    print("-" * 65)

    # ---- Example 3: a data-leaning profile ------------------------
    example_3 = ["SQL", "Data Analysis", "Statistics"]
    print(f"\nUSER INPUT #3: {example_3}")
    result_3 = recommend_roles(df, example_3, top_n=3)
    check_cold_start(result_3)
    print(result_3.to_string(index=False))
    print("-" * 65)

    # ---- Interactive mode (optional) ------------------------------
    print("\nWant to try your own skills? Run this script and pass 3+ skills")
    print("as command line arguments, e.g.:")
    print('  python3 project3_tech_stack_recommender.py Python SQL Docker\n')

    if len(sys.argv) > 3:
        user_skills = sys.argv[1:]
        print(f"USER INPUT (from CLI): {user_skills}")
        result_cli = recommend_roles(df, user_skills, top_n=3)
        check_cold_start(result_cli)
        print(result_cli.to_string(index=False))
