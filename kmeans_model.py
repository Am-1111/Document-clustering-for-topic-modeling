import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def train_kmeans(cleaned_texts, num_topics):
    texts = [" ".join(tokens) for tokens in cleaned_texts]
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    return kmeans, vectorizer, clusters

def display_kmeans_clusters(texts, clusters, num_topics):
    st.subheader("📊 KMeans Clustering Topics")
    for i in range(num_topics):
        st.write(f"**Cluster {i}**")
        examples = [texts[j] for j in range(len(clusters)) if clusters[j] == i][:3]
        for example in examples:
            st.markdown("- " + example[:200] + "...")