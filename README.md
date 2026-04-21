# 🎬 CineAI — AI Movie Recommender
### Final Year College Project

---

## 📌 Project Overview

CineAI is an AI-powered movie recommendation web application built with Python and Flask.
It uses **Content-Based Filtering** with **TF-IDF Vectorization** and **Cosine Similarity**
to recommend movies similar to any title you enter.

### 🤖 How the AI Works

1. **Data Processing** — 5,562 IMDb movies are loaded from CSV
2. **Feature Engineering** — Each movie gets a "soup" string combining:
   - Main Genre (weighted 5×)
   - Side Genres (weighted 3×)
   - Director name (weighted 2×)
   - Cast/Actors
3. **TF-IDF Vectorization** — Converts text features into a numeric matrix (shape: 5562 × 15000)
4. **Cosine Similarity** — Computes pairwise similarity between all 5562 movies
5. **Recommendation** — For any input movie, returns top-N most similar movies

---

## 📁 Project Structure

```
movie_recommender/
│
├── app.py                  ← Flask web application (main file)
│
├── data/
│   └── movies.csv          ← IMDb dataset (5562 movies)
│
├── model/
│   ├── train_model.py      ← AI model training script
│   ├── cosine_sim.pkl      ← Pre-trained similarity matrix
│   ├── title_to_idx.pkl    ← Movie title → index mapping
│   └── movies_df.pkl       ← Cleaned movie dataframe
│
├── templates/
│   ├── base.html           ← Base layout (navbar, footer)
│   ├── index.html          ← Homepage with search
│   ├── results.html        ← Recommendation results page
│   └── explore.html        ← Browse all movies
│
├── requirements.txt        ← Python dependencies
├── setup.bat               ← Windows setup script
├── setup.sh                ← Mac/Linux setup script
├── run.bat                 ← Windows run script
└── run.sh                  ← Mac/Linux run script
```

---

## 🚀 Quick Start (Step-by-Step)

### Step 1 — Install Python

Download and install Python 3.9 or higher from:
👉 https://www.python.org/downloads/

**Important (Windows):** During installation, check ✅ "Add Python to PATH"

Verify installation by opening terminal/command prompt and typing:
```
python --version
```
You should see something like: `Python 3.11.0`

---

### Step 2 — Extract the ZIP

Unzip `CineAI_MovieRecommender.zip` to any folder on your computer.
Example: `C:\Projects\CineAI\` or `~/Projects/CineAI/`

---

### Step 3 — Open Terminal in Project Folder

**Windows:**
- Open the extracted folder
- Click the address bar, type `cmd`, press Enter
- Or: Right-click inside the folder → "Open in Terminal"

**Mac/Linux:**
- Open Terminal
- Type: `cd /path/to/movie_recommender`

---

### Step 4 — Run Setup (One Time Only)

**Windows:**
```
setup.bat
```
Or double-click `setup.bat`

**Mac/Linux:**
```
chmod +x setup.sh
./setup.sh
```

This will:
- Install all required packages (Flask, pandas, scikit-learn, numpy)
- Train and save the AI model (takes ~30 seconds)

---

### Step 5 — Start the App

**Windows:**
```
python app.py
```
Or double-click `run.bat`

**Mac/Linux:**
```
python3 app.py
```
Or run `./run.sh`

---

### Step 6 — Open in Browser

Open your web browser and go to:
```
http://localhost:5000
```

🎉 **That's it! Your AI Movie Recommender is running!**

---

## 🎮 How to Use

1. **Homepage** — Type any movie title in the search bar
   - Autocomplete suggestions appear as you type
   - Choose number of recommendations (5–20)
   - Click "Find My Movies"

2. **Results Page** — See your personalised recommendations
   - Each card shows: Title, Year, Genre, Rating, Director, Cast
   - AI Match Score bar shows how similar each movie is
   - Click any recommendation to find similar movies to it

3. **Explore Page** — Browse all 5562 movies
   - Filter by genre
   - Sort by rating, year, or title
   - Click "Find similar" on any movie

---

## 📊 Dataset Information

- **Source:** IMDb Top Movies Dataset
- **Total Movies:** 5,562
- **Genres:** Action, Comedy, Drama, Crime, Biography, Animation, Adventure, Horror, Mystery, Fantasy, Western, Film-Noir, Musical
- **Fields:** Title, Year, Director, Actors, Rating, Runtime, Censor, Box Office, Genre

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.x | Core language |
| Flask 3.0 | Web framework |
| Pandas | Data processing |
| Scikit-learn | TF-IDF & Cosine Similarity |
| NumPy | Numerical operations |
| HTML/CSS/JS | Frontend UI |
| Pickle | Model serialization |

---

## 🧠 AI/ML Concepts Used

- **Content-Based Filtering** — Recommends movies based on feature similarity
- **TF-IDF (Term Frequency–Inverse Document Frequency)** — Converts text features to numeric vectors
- **Cosine Similarity** — Measures angle between feature vectors (0 = totally different, 1 = identical)
- **Feature Engineering** — Genre weighting to improve recommendation quality

---

## ❓ Troubleshooting

**"Module not found" error:**
```
pip install flask pandas scikit-learn numpy
```

**"Model file not found" error:**
```
python model/train_model.py
```

**Port already in use:**
Edit `app.py` last line, change `port=5000` to `port=5001`

**Mac pip issues:**
```
pip3 install --break-system-packages -r requirements.txt
```

---

## 📝 Project Info

- **Type:** Final Year College Project
- **Domain:** Artificial Intelligence / Machine Learning
- **Algorithm:** Content-Based Filtering with TF-IDF + Cosine Similarity
- **Dataset:** 5,562 IMDb movies across 13 genres
