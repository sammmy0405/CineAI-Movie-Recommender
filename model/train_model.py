"""
AI Movie Recommender - Model Training Script (v2)
"""
import pandas as pd, numpy as np, pickle, os, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_gross(val):
    if isinstance(val, str):
        val = val.replace('$','').replace(',','').strip()
        if 'M' in val:
            try: return float(val.replace('M','')) * 1_000_000
            except: return 0
    return 0

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['Director'] = df['Director'].fillna('Unknown')
    df['Actors'] = df['Actors'].fillna('Unknown')
    df['side_genre'] = df['side_genre'].fillna('')
    df['main_genre'] = df['main_genre'].fillna('Unknown').str.strip()
    df['Censor'] = df['Censor'].fillna('Not Rated')
    df['Total_Gross_Num'] = df['Total_Gross'].apply(clean_gross)
    df['Director_Clean'] = df['Director'].apply(lambda x: re.sub(r'^Directors?:', '', str(x)).strip())
    df = df.reset_index(drop=True)
    return df

def create_soup(row):
    main = str(row['main_genre']).strip().replace(' ', '_').replace('-', '')
    sides = [g.strip().replace(' ', '_').replace('-', '') for g in str(row['side_genre']).split(',') if g.strip()]
    director = str(row['Director_Clean']).replace(',', ' ').strip()
    actors = str(row['Actors']).replace(',', ' ').strip()
    genre_part = ' '.join([main]*5 + sides*3)
    director_part = (director + ' ') * 2
    return f'{genre_part} {director_part} {actors}'

def train_recommender(df):
    df['soup'] = df.apply(create_soup, axis=1)
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=15000)
    matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(matrix, matrix)
    title_to_idx = pd.Series(df.index, index=df['Movie_Title'].str.lower()).to_dict()
    return tfidf, matrix, cosine_sim, title_to_idx

def get_recommendations(title, df, cosine_sim, title_to_idx, n=10):
    title_lower = title.lower().strip()
    if title_lower not in title_to_idx:
        matches = [(t, i) for t, i in title_to_idx.items() if title_lower in t]
        if not matches:
            matches = [(t, i) for t, i in title_to_idx.items() if any(w in t for w in title_lower.split() if len(w)>3)]
        if matches:
            matches.sort(key=lambda x: len(x[0]))
            title_lower = matches[0][0]
        else:
            return None, f"Movie '{title}' not found in the database."
    idx = title_to_idx[title_lower]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0]!=idx][:n]
    top_indices = [s[0] for s in scores]
    top_scores = [round(float(s[1]),4) for s in scores]
    result = df.iloc[top_indices][['Movie_Title','Year','Director_Clean','Actors','Rating','Runtime(Mins)','main_genre','side_genre','Censor','Total_Gross']].copy()
    result['similarity_score'] = top_scores
    result = result.rename(columns={'Director_Clean':'Director','Runtime(Mins)':'Runtime'})
    return result, None

def save_model(df, cosine_sim, title_to_idx, output_dir='model'):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,'cosine_sim.pkl'),'wb') as f: pickle.dump(cosine_sim,f)
    with open(os.path.join(output_dir,'title_to_idx.pkl'),'wb') as f: pickle.dump(title_to_idx,f)
    df.to_pickle(os.path.join(output_dir,'movies_df.pkl'))
    print(f"Model saved: {cosine_sim.shape} matrix, {len(title_to_idx)} titles")

if __name__ == '__main__':
    print("Training AI Movie Recommender Model...")
    df = load_and_clean_data('data/movies.csv')
    _, matrix, cosine_sim, title_to_idx = train_recommender(df)
    save_model(df, cosine_sim, title_to_idx)
    print("Done!")
