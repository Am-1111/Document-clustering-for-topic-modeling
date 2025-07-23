import streamlit as st
from gensim import corpora, models

def train_lda(cleaned_texts, num_topics):
    dictionary = corpora.Dictionary(cleaned_texts)
    corpus = [dictionary.doc2bow(text) for text in cleaned_texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model, dictionary, corpus

def display_lda_topics(lda_model, dictionary, num_words=10):
    st.subheader("🔍 Topics from LDA")
    for idx, topic in lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False):
        st.write(f"**Topic {idx}**: " + ", ".join([word for word, _ in topic]))