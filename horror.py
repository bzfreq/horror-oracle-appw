import os
import json
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from urllib.parse import quote
import datetime
import random
import time

# ----- CONFIG -----
# Get variables from environment regardless of .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
INDEX_NAME = "horror-movies"
EMBED_MODEL = "text-embedding-3-small"

# ----- CLIENTS -----
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Pinecone initialization (newer API format)
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
else:
    index = None

app = Flask(__name__, static_url_path="", static_folder=".")
CORS(app, supports_credentials=True)

# Disable caching to ensure fresh results
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Simple cache for faster responses (but we'll be careful with recommendations)
response_cache = {}

HORROR_ORACLE_PROMPT = """You are the Horror Oracle, a passionate and darkly enthusiastic horror movie expert who genuinely gets excited talking about scary films.

Your personality traits:
- You're deeply passionate about horror, almost obsessively so
- You use colorful, evocative language with horror-themed metaphors
- You get genuinely excited when discussing classic horror or hidden gems

Your knowledge base:
- You know about classic horror movies, modern films, and obscure or cult classics.
- You have access to real-time movie release information (from TMDB).

Instructions:
- When asked a question, respond in character as the Horror Oracle.
- If you can't find a movie, state that your "powers are failing" but suggest a few other, similar movies to check out.
- For specific movie queries, use the provided movie data.
- For general questions (e.g., "What is your favorite horror film?"), generate a fun, in-character response.
- Do not make up movie details. If you can't find a detail, be honest about it.
- Use emojis and a playful, spooky tone.
"""

def get_movie_details(title):
    """Fetches movie details from OMDb and TMDb."""
    if not OMDB_API_KEY or not TMDB_API_KEY:
        raise Exception("OMDb or TMDb API key is not set.")

    omdb_url = f"http://www.omdbapi.com/?t={quote(title)}&apikey={OMDB_API_KEY}&plot=full"
    tmdb_search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={quote(title)}"

    # Check OMDb first
    omdb_response = requests.get(omdb_url)
    omdb_data = omdb_response.json()

    if omdb_data.get("Response") == "True":
        imdb_id = omdb_data.get("imdbID")
        return {
            "title": omdb_data.get("Title"),
            "year": omdb_data.get("Year"),
            "director": omdb_data.get("Director"),
            "plot": omdb_data.get("Plot"),
            "poster": omdb_data.get("Poster"),
            "imdb_id": imdb_id,
            "genres": omdb_data.get("Genre", "").split(", ")
        }
    
    # If OMDb fails, try TMDb for basic details and poster
    tmdb_response = requests.get(tmdb_search_url)
    tmdb_data = tmdb_response.json()

    if tmdb_data and tmdb_data.get("results"):
        result = tmdb_data["results"][0]
        # Get genres from TMDb
        genres = []
        if result.get('genre_ids'):
            # Fetch genre names from TMDb
            genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}"
            genre_response = requests.get(genre_url)
            genre_data = genre_response.json()
            genre_map = {g['id']: g['name'] for g in genre_data.get('genres', [])}
            genres = [genre_map.get(g_id, "Unknown") for g_id in result.get('genre_ids')]
            
        return {
            "title": result.get("title"),
            "year": result.get("release_date").split("-")[0] if result.get("release_date") else "N/A",
            "plot": result.get("overview"),
            "poster": f"https://image.tmdb.org/t/p/w500{result.get('poster_path')}" if result.get('poster_path') else None,
            "imdb_id": None,  # TMDb search doesn't always provide this
            "genres": genres
        }

    return None

def get_recent_horror_releases(limit=5):
    """Fetches recent horror movie releases from TMDb."""
    if not TMDB_API_KEY:
        raise Exception("TMDb API key is not set.")
    
    # Avoid caching by adding random parameter
    random_param = random.randint(10000, 99999)
    today = datetime.date.today()
    one_month_ago = today - datetime.timedelta(days=30)
    
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres=27&primary_release_date.gte={one_month_ago.isoformat()}&sort_by=primary_release_date.desc&page=1&random={random_param}"
    response = requests.get(url)
    data = response.json()
    
    releases = []
    if data and data.get("results"):
        for movie in data["results"][:limit]:
            releases.append({
                "title": movie.get("title"),
                "release_date": movie.get("release_date"),
                "poster": f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None
            })
    return releases

def get_recommendations_for_movie(movie_title):
    """Gets movie recommendations from TMDb."""
    if not TMDB_API_KEY:
        raise Exception("TMDb API key is not set.")
    
    # Add random parameter to prevent caching
    random_param = str(int(time.time()))
    
    # First search for the movie to get its ID
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={quote(movie_title)}&include_adult=true&random={random_param}"
    search_response = requests.get(search_url)
    search_data = search_response.json()
    
    if not search_data.get("results"):
        return []
        
    movie_id = search_data["results"][0]["id"]
    
    # Get recommendations
    rec_url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={TMDB_API_KEY}&random={random_param}"
    rec_response = requests.get(rec_url)
    rec_data = rec_response.json()
    
    # If no recommendations from recommendations endpoint, try similar movies
    if not rec_data.get("results") or len(rec_data.get("results", [])) < 3:
        similar_url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar?api_key={TMDB_API_KEY}&random={random_param}"
        similar_response = requests.get(similar_url)
        similar_data = similar_response.json()
        if similar_data.get("results"):
            rec_data = similar_data
    
    # Fetch genre data once to avoid multiple API calls
    genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&random={random_param}"
    genre_response = requests.get(genre_url)
    genre_data = genre_response.json()
    genre_map = {g['id']: g['name'] for g in genre_data.get('genres', [])}
    
    recommendations = []
    if rec_data and rec_data.get("results"):
        for movie in rec_data["results"][:10]:  # Get more recommendations
            # Get genres
            genres = []
            if movie.get('genre_ids'):
                genres = [genre_map.get(g_id, "Unknown") for g_id in movie.get('genre_ids')]
                # Only include horror movies or movies with few genres listed
                if "Horror" in genres or len(genres) <= 2:
                    # Get IMDb ID for buy links
                    movie_details_url = f"https://api.themoviedb.org/3/movie/{movie.get('id')}?api_key={TMDB_API_KEY}&append_to_response=external_ids&random={random_param}"
                    movie_details_response = requests.get(movie_details_url)
                    movie_details = movie_details_response.json()
                    imdb_id = movie_details.get("external_ids", {}).get("imdb_id", "")
                    
                    recommendations.append({
                        "title": movie.get("title"),
                        "year": movie.get("release_date").split("-")[0] if movie.get("release_date") else "N/A",
                        "plot": movie.get("overview"),
                        "poster_url": f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None,
                        "genres": genres,
                        "imdb_id": imdb_id,
                        "amazon_link": f"https://www.amazon.com/s?k={quote(movie.get('title'))}+movie",
                        "timestamp": str(int(time.time()))  # Add timestamp to prevent caching
                    })
                    
                    # Break once we have 5 recommendations
                    if len(recommendations) >= 5:
                        break
    
    # If we still don't have enough recommendations, try popular horror movies
    if len(recommendations) < 3:
        popular_url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres=27&sort_by=popularity.desc&random={random_param}"
        popular_response = requests.get(popular_url)
        popular_data = popular_response.json()
        
        if popular_data and popular_data.get("results"):
            for movie in popular_data["results"][:5]:
                # Don't add duplicates
                if any(r["title"] == movie.get("title") for r in recommendations):
                    continue
                    
                genres = []
                if movie.get('genre_ids'):
                    genres = [genre_map.get(g_id, "Unknown") for g_id in movie.get('genre_ids')]
                
                recommendations.append({
                    "title": movie.get("title"),
                    "year": movie.get("release_date").split("-")[0] if movie.get("release_date") else "N/A",
                    "plot": movie.get("overview"),
                    "poster_url": f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None,
                    "genres": genres,
                    "amazon_link": f"https://www.amazon.com/s?k={quote(movie.get('title'))}+movie",
                    "timestamp": str(int(time.time()))  # Add timestamp to prevent caching
                })
                
                # Stop once we have 5 recommendations
                if len(recommendations) >= 5:
                    break
                    
    # Add a "random" factor to shuffle recommendations
    random.shuffle(recommendations)
        
    return recommendations[:5]  # Limit to 5 recommendations

def get_price_comparison(imdb_id):
    """Placeholder for fetching price data."""
    return {
        "buy": f"Buy this movie for $14.99!",
        "rent": f"Rent this movie for $4.99!",
        "stream": f"Stream this movie on a subscription service!"
    }

def generate_interesting_fact(movie_title):
    """Generates an interesting fact about the movie using OpenAI."""
    if not client:
        return "The Oracle knows many secrets about this film, but cannot reveal them right now..."
        
    try:
        # Add randomness to ensure different facts each time
        random_prompt = f"Provide one interesting, lesser-known fact about the horror movie '{movie_title}'. Keep it to 1-2 sentences maximum. Make it different from previous facts you may have given. Seed: {random.randint(1, 1000)}"
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a horror movie expert. Provide ONE interesting, lesser-known fact about the specified movie. Keep it to 1-2 sentences maximum."},
                {"role": "user", "content": random_prompt}
            ],
            max_tokens=100,
            temperature=0.8
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating interesting fact: {e}")
        return "The Oracle sees dark secrets about this film, but they are shrouded in mystery..."

def check_api_keys():
    """Prints the status of all API keys on startup."""
    print("--- API Key Status ---")
    print(f"OPENAI_API_KEY: {'FOUND' if OPENAI_API_KEY else 'MISSING'}")
    print(f"PINECONE_API_KEY: {'FOUND' if PINECONE_API_KEY else 'MISSING'}")
    print(f"OMDB_API_KEY: {'FOUND' if OMDB_API_KEY else 'MISSING'}")
    print(f"TMDB_API_KEY: {'FOUND' if TMDB_API_KEY else 'MISSING'}")
    print("----------------------")


# ----- ROUTES -----
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/ask-oracle", methods=["POST"])
def ask_oracle():
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Generate a unique request ID to avoid caching
    request_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
    
    # Check if the query is asking about a specific movie
    movie_details = None
    try:
        # Simple heuristic to guess if the user is asking about a movie
        words = query.split()
        if len(words) < 5 or any(keyword in query.lower() for keyword in ["movie", "film", "watch", "see", "about"]):
            # Try to extract a movie title
            potential_title = query
            for prefix in ["tell me about", "what is", "do you know", "have you seen"]:
                if query.lower().startswith(prefix):
                    potential_title = query[len(prefix):].strip()
                    break
                    
            movie_details = get_movie_details(potential_title)
    except Exception as e:
        print(f"Failed to fetch movie details: {e}")
        # Continue without movie details
    
    # Format response based on whether movie details were found
    if movie_details:
        movie_title = movie_details.get("title")
        
        # Always get fresh recommendations for each request
        recommendations = get_recommendations_for_movie(movie_title)
        
        # Generate an interesting fact
        interesting_fact = generate_interesting_fact(movie_title)
        
        # Create a summary
        summary = f"🎬 **{movie_title}** ({movie_details.get('year')})\n\nDirected by {movie_details.get('director')}\n\n{movie_details.get('plot')}"
        
        # Create Amazon link
        amazon_link = f"https://www.amazon.com/s?k={quote(movie_title)}+movie"
        
        response_data = {
            "movie_title": movie_title,
            "summary": summary,
            "interesting_fact": interesting_fact,
            "recommendations": recommendations,
            "amazon_link": amazon_link,
            "ebay_link": f"https://www.ebay.com/sch/i.html?_nkw={quote(movie_title)}+movie",
            "imdb_id": movie_details.get("imdb_id", ""),
            "year": movie_details.get("year", ""),
            "director": movie_details.get("director", ""),
            "poster": movie_details.get("poster", ""),
            "request_id": request_id,  # Add unique request ID
            "timestamp": int(time.time())  # Add timestamp
        }
        
        # Add Cache-Control headers to response
        response = jsonify(response_data)
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        
        return response
    else:
        # If no movie details found, use OpenAI to generate a response
        if not client:
            return jsonify({"error": "OpenAI API key is missing. The Oracle's powers are failing! 👻"}), 500
            
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": HORROR_ORACLE_PROMPT},
                    {"role": "user", "content": query}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            oracle_response = response.choices[0].message.content.strip()
            
            return jsonify({
                "summary": oracle_response,
                "interesting_fact": None,
                "recommendations": [],
                "request_id": request_id,
                "timestamp": int(time.time())
            })
        except Exception as e:
            print(f"Error generating response: {e}")
            return jsonify({"error": str(e)}), 500

@app.route("/price-compare", methods=["POST"])
def price_compare():
    data = request.json
    imdb_id = data.get("imdb_id", "")
    
    if not imdb_id:
        return jsonify({"error": "IMDb ID is required"}), 400
    
    try:
        price_data = get_price_comparison(imdb_id)
        return jsonify(price_data)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/recent-releases", methods=["GET"])
def recent_releases():
    try:
        limit = request.args.get("limit", 5, type=int)
        releases = get_recent_horror_releases(limit)
        
        # Add timestamp to prevent caching
        response_data = {
            "releases": releases,
            "timestamp": int(time.time())
        }
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# Add debug route to test API connectivity
@app.route("/debug", methods=["GET"])
def debug():
    debug_info = {
        "timestamp": time.time(),
        "api_keys": {
            "OPENAI_API_KEY": bool(OPENAI_API_KEY),
            "PINECONE_API_KEY": bool(PINECONE_API_KEY),
            "OMDB_API_KEY": bool(OMDB_API_KEY),
            "TMDB_API_KEY": bool(TMDB_API_KEY)
        }
    }
    return jsonify(debug_info)

if __name__ == "__main__":
    check_api_keys()
    print("\n🔮 Horror Oracle starting (OPTIMIZED FOR SPEED)...")
    print(f"📊 Vector DB: {'CONNECTED' if index is not None else 'DISABLED'}")
    print(f"🧠 OpenAI API: {'CONNECTED' if client is not None else 'DISABLED'}")
    print(f"🎬 OMDb API: {'CONNECTED' if OMDB_API_KEY else 'DISABLED'}")
    print(f"🎥 TMDb API: {'CONNECTED' if TMDB_API_KEY else 'DISABLED'}")
    
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get("PORT", 10000))
    
    # Bind to all interfaces (0.0.0.0) on the specified port
    app.run(host="0.0.0.0", port=port)