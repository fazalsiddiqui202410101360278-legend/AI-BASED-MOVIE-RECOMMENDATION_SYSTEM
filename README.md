# 🎬 AI-Based Movie Recommendation System

> **A Machine Learning College Project** built using Python, Streamlit, Pandas, and Scikit-Learn.  
> Uses **Content-Based Filtering** and **Cosine Similarity** to recommend the top 5 most similar movies.

---

## 📌 Project Overview

This project is an interactive, web-based **Movie Recommendation System** designed for college project demonstrations and viva examinations. When a user selects or searches for their favorite movie, the system computes the mathematical similarity between the chosen film and all other movies in the dataset based on genres, plot summaries, and keywords, returning the top 5 most relevant recommendations along with posters, ratings, release years, and match percentages.

---

## 🚀 Key Features

- **Clean Dark Theme UI:** Designed with a student aesthetic using black, white, and cinema-red accents.
- **Movie Selection Dropdown:** Search or select from 50 popular movies across multiple genres (Action, Sci-Fi, Crime, Animation, Drama, Romance, Thriller).
- **Instant Recommendations:** Displays 5 similar movies in clean, responsive cards with hover effects.
- **Rich Movie Metadata:** Shows authentic movie posters, IMDb ratings, release years, genres, and storyline overviews.
- **Genuine Machine Learning:** Powered by Scikit-Learn's `CountVectorizer` and `cosine_similarity` (no hardcoded or fake recommendations).
- **Error Handling:** Displays clear warning messages if no movie is selected.
- **Offline / Low-Dependency Ready:** Dataset includes pre-verified poster links and metadata in `movies.csv` without requiring external API keys during runtime.
- **Viva Preparation Section:** Built-in interactive expander explaining the algorithm and math directly inside the web app for easy live demonstration to teachers.

---

## 🛠️ Technologies & Libraries

- **Language:** Python 3.9+
- **Frontend / Web Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pypy.org/), [NumPy](https://numpy.org/)
- **Machine Learning Library:** [Scikit-Learn](https://scikit-learn.org/)
- **Vectorization Technique:** CountVectorizer (Bag of Words Model)
- **Similarity Metric:** Cosine Similarity

---

## 🧠 How the Machine Learning Algorithm Works

The recommendation system implements **Content-Based Filtering**. Here is the 5-step pipeline:

```
[movies.csv]
     │
     ▼
[Step 1: Data Preprocessing] ──▶ Combine Genres + Overview + Keywords into 'tags'
     │
     ▼
[Step 2: Vectorization]     ──▶ CountVectorizer removes stop words & converts tags to Word Frequency Vectors
     │
     ▼
[Step 3: Similarity Matrix] ──▶ Cosine Similarity calculates angle θ between vectors
     │
     ▼
[Step 4: Ranking & Output]  ──▶ Given Movie X, sort similarity scores descending & return Top 5 matches
```

### Mathematical Formula: Cosine Similarity

Cosine similarity calculates the cosine of the angle $\theta$ between two non-zero multidimensional vectors $A$ and $B$:

$$\text{Cosine Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

- **Value = 1.0 (100%):** The two movies have identical tags.
- **Value = 0.0 (0%):** The two movies share zero common words.
- The top 5 highest scoring movies (excluding the chosen movie itself) are presented to the user.

---

## 📁 Project Directory Structure

```text
3rd Ai Project/
│
├── app.py                 # Streamlit web application & user interface
├── recommendation.py      # Content-based ML engine & cosine similarity logic
├── movies.csv             # Movie dataset (50 curated movies with posters & metadata)
├── requirements.txt       # Required Python packages
└── README.md              # Project documentation and viva Q&A
```

---

## 💻 Step-by-Step Setup & Execution Guide (VS Code)

### Step 1: Open Project in VS Code
1. Open **Visual Studio Code**.
2. Click **File > Open Folder...** and select this project directory (`d:\3rd Ai Project`).

### Step 2: Open Terminal
- Press ``Ctrl + ` `` (backtick) or go to **Terminal > New Terminal** in the top menu.

### Step 3: (Optional but Recommended) Create a Virtual Environment
```bash
python -m venv venv
```
Activate the environment:
- **Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

### Step 4: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application
```bash
streamlit run app.py
```

### Step 6: View the Website
Your web browser will automatically open:
```text
Local URL: http://localhost:8501
```
*(If it does not open automatically, hold `Ctrl` and click the `http://localhost:8501` link in your terminal).*

---

## 🎓 College Viva Questions & Answers

Be prepared to answer these common questions during your practical exam or project viva:

### Q1: What is Content-Based Filtering?
> **Answer:** Content-Based Filtering recommends items by comparing the features/attributes of items with what the user has liked in the past. In our project, movie attributes (genres, plot summary, and keywords) are compared to find similar films.

### Q2: What is the difference between Content-Based and Collaborative Filtering?
> **Answer:**
> - **Content-Based:** Looks only at the item's own metadata (e.g., genre, director, storyline). It does not need other users' ratings and does not suffer from the cold-start problem for new users.
> - **Collaborative Filtering:** Looks at user rating patterns (e.g., "Users who liked Movie A also liked Movie B"). It requires a large user database with thousands of ratings.

### Q3: Why did you use CountVectorizer instead of TF-IDF?
> **Answer:** `CountVectorizer` counts the occurrence frequency of words in the combined metadata soup (Bag of Words representation). For short text descriptions like movie genres and keywords where word occurrence directly denotes relevance, CountVectorizer gives direct, transparent, and computationally fast vectorization.

### Q4: Why Cosine Similarity instead of Euclidean Distance?
> **Answer:** Euclidean distance measures the magnitude/distance between points, which is sensitive to document length (a movie with a long synopsis would look dissimilar to one with a short synopsis). Cosine similarity measures the **direction/angle** between vectors regardless of text length, making it ideal for text comparison.

### Q5: What is the Cold-Start Problem and how does this project handle it?
> **Answer:** The Cold-Start problem occurs when a new user joins a recommendation system and has no rating history. Because our system uses Content-Based Filtering on movie attributes rather than user profiles, a user immediately gets recommendations the moment they select any movie.

---

## 👥 Student Project Details
- **Project Title:** AI-Based Movie Recommendation System
- **Domain:** Artificial Intelligence & Machine Learning
- **Framework:** Streamlit Web Application

