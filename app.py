import streamlit as st
from preprocessing import load_and_clean_data
from lda_model import train_lda, display_lda_topics
from kmeans_model import train_kmeans, display_kmeans_clusters

st.title("📚 Topic Modeling on 20 Newsgroups")
st.markdown("Apply LDA or KMeans to explore underlying topics in documents.")

# Load and preprocess
docs, cleaned_texts = load_and_clean_data()

# Choose model
model_type = st.selectbox("Choose Clustering Algorithm", ["Latent Dirichlet Allocation (LDA)", "K-Means Clustering"])

# Number of topics
num_topics = st.slider("Select Number of Topics", min_value=2, max_value=20, value=5)

# Run selected model
if st.button("Run Topic Modeling"):
    if model_type == "Latent Dirichlet Allocation (LDA)":
        lda_model, dictionary, corpus = train_lda(cleaned_texts, num_topics)
        display_lda_topics(lda_model, dictionary, num_words=10)
    else:
        kmeans, tfidf_vectorizer, clusters = train_kmeans(cleaned_texts, num_topics)
        display_kmeans_clusters(cleaned_texts, clusters, num_topics)