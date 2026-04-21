"""
AI Movie Recommender - Flask Web Application
Final Year College Project
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import re
import json

app = Flask(__name__)

# ─── Load Model ────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

def load_model():
    print("Loading AI recommendation model...")
    with open(os.path.join(MODEL_DIR, 'cosine_sim.pkl'), 'rb') as f:
        cosine_sim = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'title_to_idx.pkl'), 'rb') as f:
        title_to_idx = pickle.load(f)
    df = pd.read_pickle(os.path.join(MODEL_DIR, 'movies_df.pkl'))
    print(f"✅ Model loaded: {len(df)} movies, {len(title_to_idx)} indexed titles")
    return df, cosine_sim, title_to_idx

df, cosine_sim, title_to_idx = load_model()

# ─── Genre Colors ──────────────────────────────────────────────────────────────
GENRE_COLORS = {
    'Action':    '#e63946',
    'Comedy':    '#f4a261',
    'Drama':     '#457b9d',
    'Crime':     '#6d4c7e',
    'Biography': '#2a9d8f',
    'Animation': '#e9c46a',
    'Adventure': '#06d6a0',
    'Horror':    '#d62828',
    'Mystery':   '#7209b7',
    'Fantasy':   '#4cc9f0',
    'Western':   '#c77dff',
    'Film-Noir': '#343a40',
    'Musical':   '#fb8500',
    'Sci-Fi':    '#0077b6',
    'Thriller':  '#9d0208',
    'War':       '#606c38',
    'Sport':     '#4ade80',
    'History':   '#a98467',
    'Music':     '#ff6b6b',
    'Romance':   '#f06292',
    'Family':    '#81c784',
}

def get_genre_color(genre):
    return GENRE_COLORS.get(genre.strip(), '#6c757d')

# ─── Recommendation Engine ─────────────────────────────────────────────────────
def get_recommendations(title, n=10):
    title_lower = title.lower().strip()

    matched_title = None
    if title_lower in title_to_idx:
        matched_title = title_lower
    else:
        # Try partial match
        candidates = [(t, i) for t, i in title_to_idx.items() if title_lower in t]
        if not candidates:
            words = [w for w in title_lower.split() if len(w) > 3]
            candidates = [(t, i) for t, i in title_to_idx.items() if any(w in t for w in words)]
        if candidates:
            candidates.sort(key=lambda x: abs(len(x[0]) - len(title_lower)))
            matched_title = candidates[0][0]

    if matched_title is None:
        return None, None, f"Movie '{title}' not found. Try a different spelling or check the title."

    idx = title_to_idx[matched_title]
    source_movie = df.iloc[idx]

    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:n]

    recommendations = []
    for rank, (movie_idx, sim_score) in enumerate(scores, 1):
        row = df.iloc[movie_idx]
        sides = [g.strip() for g in str(row.get('side_genre', '')).split(',') if g.strip()]
        recommendations.append({
            'rank': rank,
            'title': row['Movie_Title'],
            'year': int(row['Year']),
            'director': str(row['Director_Clean']).strip(),
            'actors': str(row['Actors']).strip(),
            'rating': float(row['Rating']),
            'runtime': int(row['Runtime(Mins)']),
            'main_genre': str(row['main_genre']).strip(),
            'side_genres': sides,
            'censor': str(row['Censor']).strip(),
            'gross': str(row['Total_Gross']).strip(),
            'similarity': round(float(sim_score) * 100, 1),
            'genre_color': get_genre_color(str(row['main_genre']).strip()),
        })

    source_sides = [g.strip() for g in str(source_movie.get('side_genre', '')).split(',') if g.strip()]
    source_info = {
        'title': source_movie['Movie_Title'],
        'year': int(source_movie['Year']),
        'director': str(source_movie['Director_Clean']).strip(),
        'actors': str(source_movie['Actors']).strip(),
        'rating': float(source_movie['Rating']),
        'runtime': int(source_movie['Runtime(Mins)']),
        'main_genre': str(source_movie['main_genre']).strip(),
        'side_genres': source_sides,
        'censor': str(source_movie['Censor']).strip(),
        'gross': str(source_movie['Total_Gross']).strip(),
        'genre_color': get_genre_color(str(source_movie['main_genre']).strip()),
    }

    return source_info, recommendations, None

def get_all_titles():
    return sorted(df['Movie_Title'].dropna().unique().tolist())

def get_genre_stats():
    stats = df['main_genre'].value_counts().head(10).to_dict()
    return [{'genre': k, 'count': int(v), 'color': get_genre_color(k)} for k, v in stats.items()]

def get_top_movies(n=8):
    top = df.nlargest(n, 'Rating')[['Movie_Title', 'Year', 'Rating', 'main_genre', 'Director_Clean']].copy()
    result = []
    for _, row in top.iterrows():
        result.append({
            'title': row['Movie_Title'],
            'year': int(row['Year']),
            'rating': float(row['Rating']),
            'genre': str(row['main_genre']).strip(),
            'director': str(row['Director_Clean']).strip(),
            'genre_color': get_genre_color(str(row['main_genre']).strip()),
        })
    return result

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    top_movies = get_top_movies(8)
    genre_stats = get_genre_stats()
    total_movies = len(df)
    total_genres = df['main_genre'].nunique()
    return render_template('index.html',
                           top_movies=top_movies,
                           genre_stats=genre_stats,
                           total_movies=total_movies,
                           total_genres=total_genres)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        title = request.form.get('movie_title', '').strip()
        n = int(request.form.get('num_recommendations', 10))
        n = min(max(n, 3), 20)

        source, recs, error = get_recommendations(title, n=n)
        return render_template('results.html',
                               query=title,
                               source=source,
                               recommendations=recs,
                               error=error,
                               num_recs=n)
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json()
    title = data.get('title', '').strip()
    n = int(data.get('n', 10))
    source, recs, error = get_recommendations(title, n=n)
    if error:
        return jsonify({'error': error}), 404
    return jsonify({'source': source, 'recommendations': recs})

@app.route('/api/autocomplete')
def autocomplete():
    query = request.args.get('q', '').lower().strip()
    if len(query) < 2:
        return jsonify([])
    titles = df['Movie_Title'].dropna().unique()
    matches = [t for t in titles if query in t.lower()]
    matches.sort(key=lambda x: (not x.lower().startswith(query), len(x)))
    return jsonify(matches[:12])

@app.route('/api/stats')
def stats():
    return jsonify({
        'total_movies': int(len(df)),
        'total_genres': int(df['main_genre'].nunique()),
        'avg_rating': round(float(df['Rating'].mean()), 2),
        'top_genres': get_genre_stats(),
        'year_range': [int(df['Year'].min()), int(df['Year'].max())],
    })

@app.route('/explore')
def explore():
    genre_filter = request.args.get('genre', '')
    sort_by = request.args.get('sort', 'rating')
    page = int(request.args.get('page', 1))
    per_page = 24

    filtered = df.copy()
    if genre_filter:
        filtered = filtered[filtered['main_genre'].str.strip() == genre_filter]

    if sort_by == 'rating':
        filtered = filtered.sort_values('Rating', ascending=False)
    elif sort_by == 'year':
        filtered = filtered.sort_values('Year', ascending=False)
    elif sort_by == 'title':
        filtered = filtered.sort_values('Movie_Title')

    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = filtered.iloc[start:end]

    movies = []
    for _, row in page_data.iterrows():
        movies.append({
            'title': row['Movie_Title'],
            'year': int(row['Year']),
            'rating': float(row['Rating']),
            'genre': str(row['main_genre']).strip(),
            'director': str(row['Director_Clean']).strip(),
            'runtime': int(row['Runtime(Mins)']),
            'genre_color': get_genre_color(str(row['main_genre']).strip()),
        })

    genres = sorted(df['main_genre'].str.strip().unique().tolist())
    total_pages = (total + per_page - 1) // per_page

    return render_template('explore.html',
                           movies=movies,
                           genres=genres,
                           selected_genre=genre_filter,
                           sort_by=sort_by,
                           page=page,
                           total_pages=total_pages,
                           total=total)

if __name__ == '__main__':
    print("🎬 AI Movie Recommender starting...")
    print("🌐 Open: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
