# Project 3: AI Recommendation Logic — Tech Stack Recommender

**DecodeLabs Artificial Intelligence Internship — Industrial Training Kit (Batch 2026)**

## Overview

This project builds a **Content-Based Filtering** recommendation engine —
the "Digital Matchmaker" described in the training material — that maps a
user's raw skills / interests to the most relevant **job roles / tech
stacks**, using **TF-IDF vectorization** and **Cosine Similarity**.

Unlike collaborative filtering (which needs historical user behavior),
content-based filtering matches purely on item attributes, so it works
immediately with no user history required.

## Goal

Build a simple recommendation system based on user preferences:
- Take user input (skills / interests) — minimum 3 inputs
- Match preferences to items using similarity logic
- Display the top recommended items (job roles)

## Files

```
project3_tech_stack_recommender.py   -> Full end-to-end pipeline (run this)
raw_skills.csv                       -> Dataset: 15 job roles + their required skills
README.md                            -> This file
```

## The IPO Architecture (as taught in the slides)

| Phase       | What happens |
|-------------|--------------|
| **Input**   | User provides ≥3 skills (e.g. `["Python", "Cloud", "Automation"]`) |
| **Process** | Skills + job-role tags are converted into **TF-IDF weighted vectors** in a shared vocabulary space, then compared using **Cosine Similarity** |
| **Output**  | Results are sorted by score (descending) and **filtered to the Top-3** matches |

### The 4-step ranking pipeline

1. **Ingestion** — capture the user's raw skills (≥3 required).
2. **Scoring** — vectorize the user profile and every job role with TF-IDF, then compute cosine similarity between the user vector and every item vector.
3. **Sorting** — rank all job roles by similarity score, highest first.
4. **Filtering** — truncate to the Top-N (default: 3) to avoid choice overload.

### Why TF-IDF + Cosine Similarity (not raw counts / Euclidean distance)?

- **TF-IDF** downweights generic, high-frequency skills (e.g. "Git", which appears in almost every role) and upweights rare, specific skills (e.g. "Terraform", "MLOps") — so specific tags carry more signal than common ones.
- **Cosine Similarity** measures the *angle* between vectors rather than their raw magnitude, so a user with only 3 skills can still score highly against a job role with 9 listed skills, as long as the ones they share are the *distinctive* ones. Euclidean distance would unfairly penalize shorter profiles.

## Dataset: `raw_skills.csv`

15 job roles, each with 8–9 representative skill tags, covering a broad
range of tech careers: Data Scientist, DevOps Engineer, Backend/Frontend/
Full Stack Developer, Cloud Architect, ML Engineer, Data Engineer,
Cybersecurity Analyst, Mobile App Developer, DBA, SRE, BI Analyst, QA
Automation Engineer, and Systems Administrator.

## How to Run

Requirements: Python 3.8+, `pandas`, `scikit-learn`

```bash
pip install pandas scikit-learn
python3 project3_tech_stack_recommender.py
```

This runs 3 built-in demo profiles and prints their Top-3 matches.

### Try your own skills (CLI mode)

```bash
python3 project3_tech_stack_recommender.py Python SQL Docker
```

Pass 3 or more skills as arguments to get a personalized Top-3 recommendation.

## Sample Results

**Input:** `["Python", "Cloud", "Automation"]`

| Job Role | Match % |
|---|---|
| DevOps Engineer | 32.8% |
| Site Reliability Engineer | 31.6% |
| Cloud Architect | 30.9% |

**Input:** `["JavaScript", "React", "HTML"]`

| Job Role | Match % |
|---|---|
| Full Stack Developer | 51.3% |
| Frontend Developer | 51.2% |
| DevOps Engineer | 0.0% |

**Input:** `["SQL", "Data Analysis", "Statistics"]`

| Job Role | Match % |
|---|---|
| Data Scientist | 53.1% |
| Business Intelligence Analyst | 49.3% |
| Backend Developer | 10.4% |

## Handling the Cold Start Problem

If a user's skills share **zero vocabulary overlap** with the dataset
(e.g. typos, or skills not in the corpus), all similarity scores come
back as 0.0 — this is the classic **Cold Start** problem. The script
detects this and prints a notice suggesting the user try more standard
skill names. In a production system, this would typically be handled
with an onboarding survey, trending/popular fallbacks, or metadata
inference, as covered in the slides.

## Key Concepts Demonstrated

- **Content-based filtering** vs. collaborative filtering
- **Vector mapping** — converting qualitative skills into numerical arrays in a shared vocabulary space
- **TF-IDF weighting** — rewarding specific/rare tags, penalizing generic ones
- **Cosine similarity** — angle-based matching, invariant to profile size
- **4-step ranking pipeline** — Ingestion → Scoring → Sorting → Filtering
- **Cold Start problem** — and a basic mitigation strategy

## Possible Extensions

- Let users rate/weight their skills (not just list them) for a richer profile vector.
- Add a proper onboarding survey or trending-roles fallback for true cold-start users.
- Swap in real job-posting data (via a scraped or public dataset) instead of the curated sample.
- Compare content-based results against a simple collaborative-filtering baseline.

---
**DecodeLabs** · www.decodelabs.tech · decodelabs.tech@gmail.com
