import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
import gensim
import pandas as pd


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean_text(text):
    cleaned_text = " ".join([lemma.lemmatize(word) for word in text.lower().split() if word not in stop and word not in exclude])
    return cleaned_text

def perform_topic_modeling(text, num_topics=3, passes=50):
    cleaned_text = clean_text(text)
    doc = [cleaned_text.split()]
    dictionary = corpora.Dictionary(doc)
    doc_matrix = [dictionary.doc2bow(d) for d in doc]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=doc_matrix, id2word=dictionary, num_topics=num_topics, passes=passes)
    topics = lda_model.print_topics(num_topics=num_topics, num_words=10)
    topics_formatted = []
    for topic_num, topic_words in topics:
        topic_keywords = [word for word, _ in lda_model.show_topic(topic_num)][:5]
        topics_formatted.append((f"Topic {topic_num + 1}:", topic_keywords))
    return topics_formatted

import re

def generate_topic_title(topic_keywords):
    main_keywords = [re.sub(r'[^\w\s]', '', word) for word in topic_keywords[:3] if word.strip()]
    return " ".join(main_keywords).capitalize()



def main():
    st.title("Text Topic Modeling App")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.sidebar.header("Enter Text")
    input_type = st.sidebar.radio("Choose Input Type", ("Written Input", "Upload Text File"))

    if input_type == "Written Input":
        text = st.text_area("Enter your text here", "")
    else:
        uploaded_file = st.file_uploader("Upload Text File", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
        else:
            st.warning("Please upload a text file.")
            return

    if st.button("Generate Topics"):
        st.markdown("<h3>Input Text:</h3>", unsafe_allow_html=True)
        st.write(text)

        topics = perform_topic_modeling(text)
        st.markdown("<h3>Generated Topics:</h3>", unsafe_allow_html=True)
        for topic_num, topic_keywords in topics:
            topic_title = generate_topic_title(topic_keywords)
            st.write(topic_title, topic_keywords)

        st.markdown("<h3>What is Topic Modeling?</h3>", unsafe_allow_html=True)
        st.write("Topic modeling is a technique used to automatically identify topics present in a text corpus. \
            Latent Dirichlet Allocation (LDA) is one of the popular algorithms for topic modeling. \
            It works by assuming that each document is a mixture of various topics and each word in the document is attributable to one of the document's topics.")

        st.markdown("<h3>How are Topics Generated?</h3>", unsafe_allow_html=True)
        st.write("In this app, LDA is applied to the input text to identify the main themes or topics. \
            Each topic is represented by a set of keywords, which are the most relevant words for that topic.")

if __name__ == "__main__":
    main()
