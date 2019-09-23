from django.shortcuts import render,render_to_response
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import csv
from django.core.exceptions import ValidationError
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans,MiniBatchKMeans
from django.http import HttpResponse,HttpResponseRedirect
from django.urls import reverse

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def simple_upload(request):
    if request.method == 'POST' and request.FILES['csvfile']:
        try:

            csvfile = request.FILES['csvfile']
            error_message = validate_csv(csvfile)
            if error_message!='':
                print("return error message = " + error_message)
                # return HttpResponseRedirect(request.path)
                context = {
                    'errormsg' : error_message,
                    'downloadurl' : '',
                }
                return render_to_response("upload.html", context)
            km_data = request.POST.dict()

            fs = FileSystemStorage()
            filename = fs.save(csvfile.name, csvfile)
            # uploaded_file_url = fs.url(filename)
            # print("after saving file..." + filename)
            path = fs.path(filename)
            context = process_kmeans(csvfile.name,path,km_data)
            uploaded_file_url = settings.VHOST_PARAM + settings.MEDIA_URL + filename
            # print("file finished... csv created=" + uploaded_file_url)
            context['downloadurl'] = uploaded_file_url
            return render_to_response("success.html", context)
        except Exception as  e:
            print(str(e))
        # Render the HTML template index.html with the data in the context variable
        # return render(request, 'welcome/upload.html', context=context)

    return render(request,'upload.html',{})

def showlog(request):
    error_message = context['error_message']
    print("return error message = " + error_message)
    return render(request, 'upload.html', {
            'error_message': error_message
    })

def validate_csv(value):
    msg = ''
    if not value.name.endswith('.csv'):
        msg = 'Invalid file type'

    if value.size > 25000000:
        msg = 'The file is too big'
    return msg

def lemmatizeSentence(sentence):
    word_list = nltk.word_tokenize(sentence)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output



def pre_process(text):
    # lowercase
    text=text.lower()
    #remove tags
    text=re.sub("</?.*?>","",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    # remove 1 or 2 chars short word
    shortword = re.compile(r'\W*\b\w{1,2}\b')
    text=shortword.sub("",text)
    text = lemmatizeSentence(text)
    return text

#tokenizer and stemmer which returns the set of stems in the text that it is passed
def tokenize_and_stem(text):
    # first tokenizing by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filtering out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def process_kmeans(name,path,km_data):
        try:
            context = {}
            textcol = int(km_data.get("textcol"))
            numcluster = int(km_data.get("ntop"))
            ng = int(km_data.get("ng"))
            sw = km_data.get("sw")
            swlist = sw.split(',')
            stop = set(stopwords.words('english'))  
            for i in swlist:
                stop.update(i)
            filepath = os.path.join(settings.BASE_DIR, path)
            source_df=pd.read_csv(filepath)
            #Seperate Hashtags and titles to lists
            source_list = source_df.iloc[:,textcol-1].tolist()
            # print(source_df.head(5))
            corpus=[]
            for item in source_list:
                x = pre_process(item)
                corpus.append(item)

            tfidf_vectorizer = TfidfVectorizer(max_df= 0.8, max_features=150000,
                                        min_df= 1, stop_words=stop,
                                        use_idf=True, ngram_range=(1,ng))

            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
            terms = tfidf_vectorizer.get_feature_names()


            km = KMeans(n_clusters=numcluster,init='k-means++', max_iter=100, n_init=3, algorithm='auto')
            km.fit(tfidf_matrix)
    
            clusters = km.labels_.tolist()
            #sort cluster centers by proximity to centroid
            order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

            k = []
            topicIndex = []
            
            for group in set(clusters):
                # print("\nGroup : ",group, "\n-------------------")
                # print("Cluster %d words:" % group, end='')
                topicIndex.append(group)
                keywords =''
                for ind in order_centroids[group, :10]: 
                    # print(' %s' % terms[ind], end='|')
                    keywords += str(terms[ind]) +'|'
                k.append(keywords)

            # add index start from 1 instead of 0 , user requested 
            for i in range(len(clusters)): 
                clusters[i] = clusters[i] + 1
            
            for j in range(len(topicIndex)): 
                topicIndex[j] = topicIndex[j] + 1
            
            topic = pd.DataFrame({'Topic':clusters})
            numObs = topic['Topic'].value_counts()
            name=re.sub(".csv","",name)
            
            dfdata =[]
            for z in range(len(topicIndex)): 
                obj = {
                    "TopixIndex":topicIndex[z],  
                    "NumObs":numObs[z+1],   #offset 1 because index start with 1 instead of 0
                    "Keyword":k[z]
                }
                dfdata.append(obj)
            
            df = pd.DataFrame(dfdata)

            outputdf = pd.concat([source_df,topic,df], ignore_index=False, axis=1)
            outputdf.to_csv(filepath,index=False)
            context = {
                'errormsg' : '',
                'filename' : name,
                'clusters'    : dfdata,
            }
        except Exception as  e:
            print(e)

        return context 
# Create your views here.

