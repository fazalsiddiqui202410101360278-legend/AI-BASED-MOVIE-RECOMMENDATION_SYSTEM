"""
====================================================================
           AI-BASED MOVIE RECOMMENDATION SYSTEM
====================================================================
A College Project developed with Python, Streamlit, and Machine Learning.
Technique: Content-Based Filtering with CountVectorizer & Cosine Similarity.
====================================================================
"""

import streamlit as st
import pandas as pd
import recommendation

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------
# 2. CUSTOM CSS STYLING (Simple Dark Theme: Black, White, Red Accent)
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Container */
    .header-box {
        background-color: #1A1A1A;
        border-bottom: 2px solid #E50914;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 25px;
        text-align: center;
    }
    .header-title {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    .header-title span {
        color: #E50914;
    }
    .header-subtitle {
        color: #B3B3B3;
        font-size: 1rem;
        margin-bottom: 0px;
    }
    .header-badge {
        display: inline-block;
        background-color: #2D1416;
        color: #FF4B4B;
        border: 1px solid #E50914;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-top: 10px;
        font-weight: 500;
    }

    /* Selected movie preview box */
    .selected-card {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        border-left: 4px solid #E50914;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .selected-title {
        color: #FFFFFF;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .selected-meta {
        color: #B3B3B3;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }

    /* Movie Recommendation Card */
    .movie-card {
        background-color: #1E1E1E;
        border: 1px solid #2B2B2B;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        border-color: #E50914;
    }
    .movie-card img {
        border-radius: 6px;
        width: 100%;
        height: 280px;
        object-fit: cover;
        margin-bottom: 10px;
    }
    .movie-card-title {
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 600;
        margin: 6px 0 4px 0;
        min-height: 44px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .movie-badge-row {
        display: flex;
        justify-content: center;
        gap: 6px;
        margin-bottom: 6px;
        flex-wrap: wrap;
    }
    .rating-badge {
        background-color: #262626;
        color: #FFD700;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid #3A3A3A;
    }
    .year-badge {
        background-color: #262626;
        color: #B3B3B3;
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid #3A3A3A;
    }
    .match-badge {
        background-color: #2E1517;
        color: #FF5A5F;
        border: 1px solid #E50914;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        margin-top: 4px;
        display: inline-block;
    }
    .genre-text {
        color: #8C8C8C;
        font-size: 0.78rem;
        margin-top: 4px;
        min-height: 32px;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #E50914;
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-size: 1rem;
        width: 100%;
        transition: background-color 0.2s ease, transform 0.1s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #B80710;
        color: #FFFFFF;
        transform: translateY(-2px);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777777;
        font-size: 0.85rem;
        margin-top: 50px;
        padding: 20px 0;
        border-top: 1px solid #222222;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# 3. LOAD DATASET AND COMPUTE SIMILARITY (Cached for high performance)
# ------------------------------------------------------------------
@st.cache_data
def get_dataset():
    """Load and cache the movie dataset from CSV."""
    return recommendation.load_movie_data("movies.csv")


@st.cache_data
def get_similarity_matrix(_df):
    """Compute and cache the Cosine Similarity matrix."""
    return recommendation.compute_similarity(_df)


try:
    df = get_dataset()
    similarity_matrix = get_similarity_matrix(df)
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"❌ Error loading dataset: {e}")


# ------------------------------------------------------------------
# 4. HEADER SECTION
# ------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">🎬 <span>AI</span> Movie Recommendation System</div>
    <div class="header-subtitle">A Student Machine Learning Project using Content-Based Filtering & Cosine Similarity</div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# 5. USER INTERFACE: MOVIE SELECTION
# ------------------------------------------------------------------
if data_loaded:
    # Sort movie titles alphabetically for convenient selection
    movie_list = sorted(df['title'].tolist())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Movie dropdown selection with a placeholder option
        selected_movie = st.selectbox(
            "🔍 Search or select a movie you like:",
            options=["-- Select a movie --"] + movie_list,
            index=0,
            help="Type or select a movie to get recommendations"
        )
    
    with col2:
        # Spacer for vertical alignment with dropdown
        st.write("")
        st.write("")
        recommend_clicked = st.button("🍿 Recommend Movies", type="primary")

    # --------------------------------------------------------------
    # 6. SELECTED MOVIE DETAILS (PREVIEW)
    # --------------------------------------------------------------
    if selected_movie and selected_movie != "-- Select a movie --":
        movie_info = recommendation.get_movie_details(selected_movie, df)
        if movie_info:
            st.markdown(f"""
            <div class="selected-card">
                <div class="selected-title">📌 Selected Movie: <b>{movie_info['title']}</b> ({movie_info['release_year']})</div>
                <div class="selected-meta">
                    <span style="color: #FFD700;">★ {movie_info['rating']} / 10</span> &nbsp;|&nbsp; 
                    <span>🎭 {movie_info['genres']}</span>
                </div>
                <div style="color: #D1D1D1; font-size: 0.9rem; line-height: 1.4;">
                    <b>Plot Overview:</b> {movie_info['overview']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------
    # 7. RECOMMENDATION LOGIC & RESULTS DISPLAY
    # --------------------------------------------------------------
    if recommend_clicked:
        # Check if the user selected a valid movie
        if not selected_movie or selected_movie == "-- Select a movie --":
            st.warning("⚠️ Movie not found or not selected! Please select a movie from the dropdown to get recommendations.")
        else:
            with st.spinner("Finding similar movies using Cosine Similarity..."):
                recommendations = recommendation.get_recommendations(
                    selected_movie, df, similarity_matrix, top_n=5
                )
            
            if recommendations:
                st.markdown(f"### 🎯 Top 5 Recommended Movies for *{selected_movie}*:")
                
                # Create 5 equal columns for the 5 recommended movie cards
                cols = st.columns(5)
                
                for idx, col in enumerate(cols):
                    if idx < len(recommendations):
                        movie = recommendations[idx]
                        with col:
                            # Use Streamlit native container with cards
                            with st.container():
                                # Display movie poster
                                poster_url = movie.get('poster_url')
                                if poster_url:
                                    st.image(poster_url, use_container_width=True)
                                else:
                                    # Fallback placeholder if no poster URL
                                    st.image("https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                                
                                # Movie Title
                                st.markdown(f"**{movie['title']}**")
                                
                                # Metadata: Rating & Year
                                st.markdown(
                                    f"<div class='movie-badge-row'>"
                                    f"<span class='rating-badge'>★ {movie['rating']}</span>"
                                    f"<span class='year-badge'>{movie['release_year']}</span>"
                                    f"</div>", 
                                    unsafe_allow_html=True
                                )
                                
                                # Similarity score badge
                                st.markdown(
                                    f"<div style='text-align: center; margin-bottom: 6px;'>"
                                    f"<span class='match-badge'>{movie['similarity_score']}% Match</span>"
                                    f"</div>", 
                                    unsafe_allow_html=True
                                )
                                
                                # Genres
                                st.caption(f"🎭 {movie['genres']}")
                                
                                # Short plot preview in an expander
                                with st.expander("📖 Storyline"):
                                    st.write(movie['overview'])
            else:
                st.error("⚠️ No recommendations found. Please try another movie.")

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 9. FOOTER
# ------------------------------------------------------------------
st.markdown("""
<div class="footer">
    🎬 AI Movie Recommendation System • Machine Learning Project<br>
    Built with <b>Python</b>, <b>Streamlit</b>, <b>Pandas</b>, and <b>Scikit-Learn</b>
</div>
""", unsafe_allow_html=True)

