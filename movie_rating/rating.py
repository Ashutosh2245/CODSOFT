import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset with encoding
df = pd.read_csv("movies.csv", encoding='ISO-8859-1')

# Data Cleaning
df = df.dropna(subset=['Name', 'Rating'])
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)
df = df.dropna(subset=['Rating', 'Votes'])

# --- Functions ---

def get_top_movies_by_rating(n=10):
    return df.sort_values(by='Rating', ascending=False).head(n)

def get_top_movies_by_votes(n=10):
    return df.sort_values(by='Votes', ascending=False).head(n)

def search_by_actor(actor_name):
    mask = df[['Actor 1', 'Actor 2', 'Actor 3']].apply(lambda x: x.str.contains(actor_name, case=False, na=False)).any(axis=1)
    return df[mask][['Name', 'Rating', 'Votes']]

def get_genre_distribution():
    all_genres = df['Genre'].dropna().str.split(', ').explode()
    return all_genres.value_counts()

def get_average_rating_by_genre():
    genre_df = df.copy()
    genre_df = genre_df.dropna(subset=['Genre'])
    genre_df['Genre'] = genre_df['Genre'].str.split(', ')
    genre_df = genre_df.explode('Genre')
    return genre_df.groupby('Genre')['Rating'].mean().sort_values(ascending=False)

# --- Display Info ---

top_rated = get_top_movies_by_rating()
print("🎬 Top Rated Movies:\n", top_rated[['Name', 'Year', 'Genre', 'Rating']])

top_voted = get_top_movies_by_votes()
print("\n📊 Most Voted Movies:\n", top_voted[['Name', 'Year', 'Votes']])

actor_results = search_by_actor("Irrfan Khan")
print("\n🎭 Movies featuring Irrfan Khan:\n", actor_results)

# --- Visualization ---

# Top Rated Movies
plt.figure(figsize=(10, 6))
sns.barplot(data=top_rated, y='Name', x='Rating', hue='Name', palette='viridis', legend=False)
plt.title('Top 10 Rated Movies')
plt.xlabel('Rating')
plt.ylabel('Movie')
plt.tight_layout()
plt.show()

# Top Voted Movies
plt.figure(figsize=(10, 6))
sns.barplot(data=top_voted, y='Name', x='Votes', hue='Name', palette='magma', legend=False)
plt.title('Top 10 Most Voted Movies')
plt.xlabel('Votes')
plt.ylabel('Movie')
plt.tight_layout()
plt.show()

# Genre Distribution
genre_dist = get_genre_distribution()
plt.figure(figsize=(12, 6))
genre_dist[:15].plot(kind='bar', color='coral')
plt.title('Top 15 Most Common Genres')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()

# Average Rating by Genre
avg_rating_genre = get_average_rating_by_genre()
plt.figure(figsize=(12, 6))
avg_rating_genre[:15].plot(kind='bar', color='seagreen')
plt.title('Top 15 Genres by Average Rating')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.show()
