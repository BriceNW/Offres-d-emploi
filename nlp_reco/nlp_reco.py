from flask import Flask, render_template, request

import pandas as pd

import numpy as np
import re

import nltk
# nltk.download
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from IPython.display import HTML

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)


def text_preproc(row):
    row = re.sub(r'\W+', ' ', row)  # remove non aphanumeric characters
    row = row.replace('\r', '')  # remove \r
    row = row.replace('\n', '')  # remove \n
    row = row.replace('\d+', '')  # remove numbers
    row = re.sub(r'[^\w\s]', '', row)  # remove punctuations
    row = ' '.join([item for item in word_tokenize(row) if item not in stop_words])  # tokenization
    row = row.lower()  # put all words in lowcase
    row = lemmatizer.lemmatize(row)  # lemmatization
    return row


# Load the tfidf_vectorizer fitted on all the jobs_list dataset
# tfidf_vectorizer = TfidfVectorizer()

# Load the job dataset
job_list = pd.read_csv('job_list.csv', low_memory=False)
job_list.fillna('', inplace=True)
# Job_list['Combined_info'] = job_list['Position'] + ' ' + job_list['Description']

vectorizer = open('vectorizer.pkl', 'rb')
vect = joblib.load(vectorizer)

jobs_vect = open('jobs_vectors.pkl', 'rb')
jobs_vectors = joblib.load(jobs_vect)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        work_exp = [text_preproc(request.form['work_exp'])]
        current_job = [text_preproc(request.form['current_job'])]
        current_job_desc = [text_preproc(request.form['current_job_desc'])]
        current_job_info = [text_preproc(request.form['current_job']) + ' '
                            + text_preproc(request.form['current_job_desc'])]
        poi = [text_preproc(request.form['poi'])]
        location = text_preproc(request.form['location'])
        distance = float(request.form.get('distance'))

        work_exp_vector = vect.transform(work_exp)
        current_job_vector = vect.transform(current_job_info)
        poi_job_vector = vect.transform(poi)

        # Gather all user info into a dataframe

        data = [request.form['work_exp'],
                request.form['current_job'],
                request.form['current_job_desc'],
                request.form['poi'],
                request.form['location']
                ]
        df_user = pd.DataFrame(np.array(data).reshape(-1, len(data)),
                               columns=['work experience',
                                        'current job',
                                        'job description',
                                        'Position of interest',
                                        'Location'])

        # Calculate the cosine similarity function between the user info and the jobs info

        similarity_work_exp = cosine_similarity(work_exp_vector, jobs_vectors).flatten()
        similarity_current_job = cosine_similarity(current_job_vector, jobs_vectors).flatten()
        similarity_poi = cosine_similarity(poi_job_vector, jobs_vectors).flatten()

        # Get the 10 best matches in each category

        best10_indices_work_exp = similarity_work_exp.argsort()[:-10:-1]
        similarity_scores_work_exp = similarity_work_exp[best10_indices_work_exp]
        best10_jobs_work_exp = job_list.iloc[best10_indices_work_exp, :]
        best10_jobs_work_exp['similarity coefficient'] = similarity_scores_work_exp

        best10_indices_current_job = similarity_current_job.argsort()[:-10:-1]
        similarity_scores_current_job = similarity_work_exp[best10_indices_current_job]
        best10_jobs_current_job = job_list.iloc[best10_indices_current_job, :]
        best10_jobs_current_job['similarity coefficient'] = similarity_scores_current_job

        best10_indices_poi = similarity_poi.argsort()[:-10:-1]
        similarity_scores_poi = similarity_poi[best10_indices_poi]
        best10_jobs_poi = job_list.iloc[best10_indices_poi, :]
        best10_jobs_poi['similarity coefficient'] = similarity_scores_poi

    return render_template('result.html',
                           tables=[df_user.to_html(classes='user', index=False),
                                   best10_jobs_work_exp.to_html(classes='work_exp', index=False),
                                   best10_jobs_current_job.to_html(classes='current_job', index=False),
                                   best10_jobs_poi.to_html(classes='poi', index=False)],
                           titles=['na',
                                   'Summary of your past, current and future job details',
                                   'Recommended jobs based on previous occupied jobs',
                                   'Recommended jobs based on current job',
                                   'Recommended jobs based on position of interest'])


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
