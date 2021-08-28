import streamlit as st
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
remove= ['not']
new_stop_words = stop_words.difference(remove)
lemmatizer= WordNetLemmatizer()

 def tokenization(words)
    words = words.astype(str)
    words = words.str.lower()
    tokenized = words.str.split()
    return tokenized

 def remove_stopwords(tokenized):
    filter = [word for word in tokenized if word not in new_stop_words]
    return filter

 def lemmatization(filter):
    limmatized = [lemmatizer.lemmatize(word) for word in filter]
    return limmatized

 def clean_review(review)
    review = tokenization(review)
    review = remove_stopwords(review)
    review = lemmatization(review)
    review = review_c.str.join(" ")
    return review

def raw_test(review, model, vectorizer):
    # Clean Review
    review_c = clean_review(review)
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"
    
model_name = 'rf_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'
loaded_model = pickle.load(open(model_name, 'rb'))
loaded_vect = pickle.load(open(vectorizer_name, 'rb'))

def main():
    st.title("Amazon Food Review")
    text = st.text_area("Enter Your Review","")
    if st.button("Analyze"):
        res = raw_test(text, loaded_model, loaded_vect)
        st.success(res)

if __name__ == '__main__':
	main()