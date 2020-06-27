
# LABOR – Learning and Building on Reviews

**Machine Learning 2.0 Final Project**

**MSDS 2020 Learning Team 4**
* Ria Ysabelle L. Flora
* Crisanto E. Chua
* Armand Louis A. De Leon
* Jeddahlyn V. Gacera

**Asian Institute of Management**

# Summary/Abstract

Employee management is one of the most important functions within an organization. Recent studies have shown that employee perceptions of culture, current management, opportunities for growth, and other intangible factors are correlated to a company’s financial well-being. Therefore, it is very important that managers are able to develop necessary skills to effectively connect with employees. It has also given a new dimension to human resource management from being reactionary (i.e. solving employee complaints, etc.) to a more proactive role (giving insights that help in creating policies that prevent or minimize employee dissatisfaction). In this paper, we explore models that predict employee sentiment based on text gathered in employee review data from Glassdoor.com. With the models we have developed, which accurately predict employee sentiment and give insights on what push these ratings, we now are able to provide organizations a new way of better understanding their employees, via internal quarterly reviews or through employee comments in their in-house networks.

# Introduction

High employee turnover is one of the biggest challenges faced by businesses today regardless of location, size, nature or business strategy. According to a 2013 study by consumer credit reporting agency, Equifax, 40% of employees who leave their jobs do so within six months of starting a position (Paul, 2013). If this is not addressed organizational cost expenditure will proportionately increase (Ali, 2009).

In another study published in 2017, 42% of millennials expect to change jobs every 1 to 3 years, at the very least (Jobvite, 2017). However, what is more disturbing is the fact that in a survey conducted in 2017, only 9% of senior managers believe that turnover is an urgent issue (Pollock, 2017).

Online career websites such as Glassdoor.com have provided valuable information on how employees view their current and previous companies and the reasons why they think highly or poorly of them through reviews. The problem however is its qualitative nature which makes it difficult to compare the experience to other companies just based on the text review. A translation of the qualitative review to a quantitative metric can help bridge the gap. Although Glassdoor.com already has the option of adding a quantitative rating (1-5 stars, 5 being the highest), sometimes the rate chosen is not reflective of the actual sentiment of the employee.

The goal of the study is to predict quantitative ratings of companies with the use of sentiment analysis on the qualitative text reviews. The model produced could potentially help companies rate themselves accurately with internally generated text reviews or assist other company rating platforms who don’t have quantitative rating schemes transform these into numerical ratings that are based on the same metric.

# Methodology

This study is an exploration of the possibilities of creating hybrid NLP models for predictions: using both word embeddings and other numerical features to make a classification prediction. The main reason this is not possible without neural networks (specifically embedding models) is because conventional bag of words creates sparse vector representations of words. This sparse representation makes it such that adding other features to the matrix would have little to no impact.

To achieve this, the vector representation of the numerical features is added to the word vector created from word embeddings. The numerical features are probabilities resulting from topic modeling through LDA. A neural network is then trained to make predictions on whether a job review is negative, neutral, or positive.

The implementation of our models is based on two different libraries in Python. The topic modeling through LDA was done through the gensim library, which features extensive functionality for more updated NLP methods. Meanwhile, word Embedding and the Stacked GRU were implemented through Tensorflow Keras. 

## 0. Import libraries


```python
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import sqlite3
import re
import json
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import string 
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_sm')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Activation
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from sklearn.metrics import classification_report
```

## 1. Data gathering

This dataset contains company reviews from Glassdoor.com, a jobs and recruiting website that contains a database of millions of company reviews, salary reports, interview reviews, CEO approval ratings, and other information. Before reviews are posted by an employee, they must verify that they currently or previously worked at the listed company. In addition, reviews are completely anonymous and voluntarily contributed by the employee seeking for jobs in other companies in exchange of being able to have unlimited access to the website. These features of Glassdoor ensure the authenticity of the reviews and help reduce reviewer bias.

The dataset was obtained by scraping the reviews of 73 Philippine-based companies from Glassdoor.com with 3,718 unique data points. Each of the reviews contains text on the pros and cons of working for the company, advice to management and a short summary of the review. The target variable is the star rating which has a value of 1 to 5 (lowest to highest).

<img src="https://github.com/jeddahlyngacera/labor-glassdoor/blob/master/img1.PNG">
<p style="text-align: center;">Figure 1 - Sample data from Glassdoor.com</p>

### 1.1. Load target database


```python
conn = sqlite3.connect('glassdoor.db')
```

### 1.2. Define functions for scraping


```python
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/'
    '537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
```


```python
def extract_from_link(link, soup):
    '''Returns a dataframe from extracted data from link using output of 
       BeautifulSoup.'''
    
    dict_ = {}
    dict_['review_title'] = [i.text.strip('"') for i in soup.select('h2.h2 span')]
    dict_['rating'] = [i['title'] for i in soup.select('span.rating span')][1:]
    dict_['job_title'] = [i.text for i in soup.select('span.authorJobTitle')]
    dict_['main_text'] = [i.text for i in soup.select('p.mainText')]

    reviews = []
    for rev in soup.select('div.col-sm-11'):
        texts = ''
        for j in [i.select('p.strong ~ p') for i in rev]:
            if len(j)!=0:
                texts += j[0].text+'\n'
        reviews.append(texts)
    dict_['review'] = reviews

    df = pd.DataFrame.from_dict(dict_)
    df['company'] = re.findall(r'(.*?) Reviews ', soup.title.text)[0]
    df['link'] = link
    
    return df
```


```python
def extract_all(link, save_to_db=True):
    '''Main scraper function. Returns a dataframe of all data extracted from
      all possible pages of given link and appends it to the target db if
      save_to_db is True.'''
    
    df_main = pd.DataFrame(columns=['review_title', 'rating', 'job_title', 
                                    'main_text', 'review'])

    i = 1
    while i > 0:
        if i==1:
            next_link = link
        else:
            next_link = re.findall(r'(.*).htm', link)[0]+'_P'+str(i)+'.htm'
        source = requests.get(next_link, headers=headers)
        soup = BeautifulSoup(source.text, 'lxml')
        if len(soup.select('h2.h2 span'))==0: 
            break
        df_main = df_main.append(extract_from_link(next_link, soup), sort=False)
        print('Extracted', next_link)
        i += 1
        
    df_main = df_main.reset_index(drop=True)
    if save_to_db:
        df_main.to_sql('reviews_tbl', conn, if_exists='append')
        
    return df_main
```


```python
def pickup_links(links):
    '''Returns list of links to scrape by checking if given links already 
       exist in the target db.'''
    try:
        extracted_links = pd.read_sql('''SELECT DISTINCT link FROM reviews_tbl''', 
                                      conn)['link'].values
        new_links = [i for i in links if i not in extracted_links]   
    except:
        new_links = links  
    return new_links
```

### 1.3. Provide links to scrape

Note that links should follow the format: 
<pre>
https://www.glassdoor.com/Reviews/&lt;company-name-*&gt;.htm
</pre>


```python
links = ['https://www.glassdoor.com/Reviews/Edukasyon-ph-Reviews-E1378940.htm',
         'https://www.glassdoor.com/Reviews/Asticom-Technologies-Reviews-E1523794.htm',
         'https://www.glassdoor.com/Reviews/JeonSoft-Reviews-E1581997.htm',
         'https://www.glassdoor.com/Reviews/Vibal-Publishing-House-Reviews-E566841.htm',
         'https://www.glassdoor.com/Reviews/AAISI-Reviews-E751354.htm',
         'https://www.glassdoor.com/Reviews/Systems-and-Software-Consulting-Group-Reviews-E579828.htm',
         'https://www.glassdoor.com/Reviews/Dermclinic-Reviews-E624818.htm',
         'https://www.glassdoor.com/Reviews/FilAm-Software-Technology-Reviews-E1017330.htm']
```

### 1.4. Check if links already exist in the database


```python
new_links = pickup_links(links)
```


```python
new_links
```




    ['https://www.glassdoor.com/Reviews/Edukasyon-ph-Reviews-E1378940.htm',
     'https://www.glassdoor.com/Reviews/Asticom-Technologies-Reviews-E1523794.htm',
     'https://www.glassdoor.com/Reviews/JeonSoft-Reviews-E1581997.htm',
     'https://www.glassdoor.com/Reviews/Vibal-Publishing-House-Reviews-E566841.htm',
     'https://www.glassdoor.com/Reviews/AAISI-Reviews-E751354.htm',
     'https://www.glassdoor.com/Reviews/Systems-and-Software-Consulting-Group-Reviews-E579828.htm',
     'https://www.glassdoor.com/Reviews/Dermclinic-Reviews-E624818.htm',
     'https://www.glassdoor.com/Reviews/FilAm-Software-Technology-Reviews-E1017330.htm']




```python
len(new_links)
```




    8



### 1.5. Scrape data from links and save to db


```python
for link in new_links:
    extract_all(link, save_to_db=True)
```

    Extracted https://www.glassdoor.com/Reviews/Edukasyon-ph-Reviews-E1378940.htm
    Extracted https://www.glassdoor.com/Reviews/Edukasyon-ph-Reviews-E1378940_P2.htm
    Extracted https://www.glassdoor.com/Reviews/Asticom-Technologies-Reviews-E1523794.htm
    Extracted https://www.glassdoor.com/Reviews/Asticom-Technologies-Reviews-E1523794_P2.htm
    Extracted https://www.glassdoor.com/Reviews/JeonSoft-Reviews-E1581997.htm
    Extracted https://www.glassdoor.com/Reviews/JeonSoft-Reviews-E1581997_P2.htm
    Extracted https://www.glassdoor.com/Reviews/Vibal-Publishing-House-Reviews-E566841.htm
    Extracted https://www.glassdoor.com/Reviews/Vibal-Publishing-House-Reviews-E566841_P2.htm
    Extracted https://www.glassdoor.com/Reviews/AAISI-Reviews-E751354.htm
    Extracted https://www.glassdoor.com/Reviews/AAISI-Reviews-E751354_P2.htm
    Extracted https://www.glassdoor.com/Reviews/Systems-and-Software-Consulting-Group-Reviews-E579828.htm
    Extracted https://www.glassdoor.com/Reviews/Dermclinic-Reviews-E624818.htm
    Extracted https://www.glassdoor.com/Reviews/FilAm-Software-Technology-Reviews-E1017330.htm
    

### 1.6. Check if data are loaded to the db


```python
df_reviews = pd.read_sql('''SELECT * FROM reviews_tbl''', conn)
```


```python
df_reviews.shape
```




    (3718, 8)




```python
df_reviews.rating.value_counts()
```




    4.0    1119
    3.0     994
    5.0     839
    1.0     401
    2.0     365
    Name: rating, dtype: int64




```python
df_reviews.tail(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>review_title</th>
      <th>rating</th>
      <th>job_title</th>
      <th>main_text</th>
      <th>review</th>
      <th>company</th>
      <th>link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3713</td>
      <td>5</td>
      <td>Doesn't Recommend</td>
      <td>1.0</td>
      <td>Former Employee - Anonymous Employee</td>
      <td>I worked at FilAm Software Technology full-tim...</td>
      <td>Free coffee and rice.\r\nAnnual increase\nUsel...</td>
      <td>FilAm Software Technology</td>
      <td>https://www.glassdoor.com/Reviews/FilAm-Softwa...</td>
    </tr>
    <tr>
      <td>3714</td>
      <td>6</td>
      <td>'Learn on your own' company</td>
      <td>2.0</td>
      <td>Developer</td>
      <td>I worked at FilAm Software Technology</td>
      <td>Good place to start your career as dev in a ha...</td>
      <td>FilAm Software Technology</td>
      <td>https://www.glassdoor.com/Reviews/FilAm-Softwa...</td>
    </tr>
    <tr>
      <td>3715</td>
      <td>7</td>
      <td>Working Experience</td>
      <td>2.0</td>
      <td>Current Employee - Anonymous Employee</td>
      <td>I have been working at FilAm Software Technolo...</td>
      <td>We rarely have to work over time\nNo contract ...</td>
      <td>FilAm Software Technology</td>
      <td>https://www.glassdoor.com/Reviews/FilAm-Softwa...</td>
    </tr>
    <tr>
      <td>3716</td>
      <td>8</td>
      <td>needs improvement</td>
      <td>1.0</td>
      <td>Former Employee - Anonymous Employee</td>
      <td>I worked at FilAm Software Technology full-time</td>
      <td>Annual Increase\r\n10 days paid time off for t...</td>
      <td>FilAm Software Technology</td>
      <td>https://www.glassdoor.com/Reviews/FilAm-Softwa...</td>
    </tr>
    <tr>
      <td>3717</td>
      <td>9</td>
      <td>A good company</td>
      <td>5.0</td>
      <td>Applications Developer</td>
      <td>I have been working at FilAm Software Technolo...</td>
      <td>Agile, Up to date, new hire friendly\nNone I c...</td>
      <td>FilAm Software Technology</td>
      <td>https://www.glassdoor.com/Reviews/FilAm-Softwa...</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_sql('''SELECT COUNT(DISTINCT company) company_count FROM reviews_tbl''', conn)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>company_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>73</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Data preparation and selection

Text pre-processing is the first step in preparing text data for analysis. For this study, the following were done to clean the text before part of speech (POS) tagging:
1.	Lowercase all characters
2.	Remove all special characters and punctuations
3.	Fix word contractions (convert “ain’t” into “is not”)
4.	Lemmatizing and POS tagging (done simultaneously)
5.	Stop word removal

As an additional note, stop words are removed after POS tagging since the model involved in POS tagging may lose information it needs to properly tag words.

### 2.1. Load data from database


```python
conn = sqlite3.connect('glassdoor.db')
```


```python
df_reviews = pd.read_sql('''SELECT * FROM reviews_tbl''', conn)
```

### 2.2. Filter to needed columns: `review` and `rating`


```python
df = df_reviews[['review', 'rating']]
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Very accomodating staff and clean environment\...</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>well known company in the Philippines and in l...</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Well we all know that Cebuana is the country's...</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>acra acra acra acra acra\noperations division ...</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Salary is always on time\nToo Much Pressure es...</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.3. Define function for text (POS) cleaning and processing

**POS Tagging**

POS tagging allows a machine to properly identify how a certain word was used within a sentence, whether it was used as a noun, adjective, verb, etc. This has multiple applications, such as in telling a machine how to pronounce a word in text-to-speech (TTS) programs (Bellegarda, 2015), in aspect-level sentiment analysis, or automated grammar checking. POS tagging makes it possible to filter out particular parts of speech that may not be relevant to the analysis, and in a way can be used as a crude means of dimensionality reduction.

For this study, the POS tagger in the SpaCy library was used. Hence, lemmatizing is done alongside the POS tagging process. Parts of speech such as nouns, verbs, adjectives, and adverbs were retained for analysis. The study finds that reducing word usage further according to POS caused accuracy to suffer significantly.



```python
stops = stopwords.words('english') + ['work', 'good']

with open('contraction_mapping.json', 'r') as f:
    contraction_mapping = json.load(f)
```


```python
def clean_text(text, clean_only=False, 
               parts_of_speech=['ADJ' ,'NOUN', 'ADV', 'VERB'],
              remove_sw=True, sw=stops):
    """
    Cleans text and filters according to part of speech.
    
    Parameters
    ----------
    text : str
    
    clean_only : bool
        default at false, will return cleaned string with no tagging
    
    parts_of_speech : list of strings
        refer to parts of speech in SpaCy
        
    remove_sw : bool
    
    sw : list of strings
        add your own if necessary
        
    Returns
    -------
    out3 : str
        string with parts of speech filtered
    """
    # cleaning
    text = text.lower()
    text = text.replace('\xa0', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = re.sub(r'[^\w\s]+', ' ', text)
    text = re.sub("p*\d", "", text)
    text = re.sub(r" +", ' ', text)
    text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
    
    if clean_only == True:
        return text  
    
    else: 
        # pass text into nlp then remove stopwords
        text = nlp(text)

        # .lemma_ and .pos_ are helpful extracting the lemmatized
        # word and part of speech.

        out = []
        for token in text:
             out.append((token.lemma_, token.pos_))
        poss = parts_of_speech

        out3 = ''

        for item in out:
            if item[1] in poss:
                out3 = out3 + ' ' + item[0]

        if remove_sw:
            dummy = out3.split()
            dummy = [word for word in dummy if word not in sw]
            out3 = ' '.join(dummy)
            return out3.strip()

        else:
            
            return out3.strip()
```

### 2.4. Process `review` column


```python
pd.options.mode.chained_assignment = None
df['review'] = df['review'].apply(lambda x: clean_text(x))
```


```python
df.shape
```




    (3718, 2)




```python
df.head(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>accomodate staff clean environment think con c...</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>well know company line money remittance pawn j...</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>know country big company pawnshop great benefi...</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>acra division family life always render overti...</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>salary always time much pressure especially se...</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.5. Save cleaned data to a pickle file `DF_glassdoor_3718.pkl`

```python
with open('DF_glassdoor_3718.pkl', 'wb') as f:
    pickle.dump(df, f)
```

### 2.6. Binning the ratings

For the particular use case, simplifying the classes from 5 to 3 makes the numbers more interpretable. For example, it is difficult to articulate the difference of a 1 from a 2, and so in. Hence, the combining ratings 1 and 2 into “negative”, 3 into “neutral”, and 4 to 5 into “positive”. This also makes the classification problem simpler for model training.

Convert 5-star ratings to 3 classes:

 * `1`: negative
 * `2`: neutral
 * `3`: positive

|original rating | new rating |
|---|---|
|1.0 | 1.0 |
|2.0 | 1.0 |
|3.0 | 2.0 |
|4.0 | 3.0 |
|5.0 | 3.0 |


```python
df['rating'] = df['rating'].apply(lambda x: str(x))
```


```python
df.rating.value_counts()
```




    4.0    1119
    3.0     994
    5.0     839
    1.0     401
    2.0     365
    Name: rating, dtype: int64




```python
df.loc[df['rating']=='2.0', 'rating'] = '1.0'
df.loc[df['rating']=='3.0', 'rating'] = '2.0'
df.loc[df['rating']=='4.0', 'rating'] = '3.0'
df.loc[df['rating']=='5.0', 'rating'] = '3.0'
```


```python
df.rating.value_counts()
```




    3.0    1958
    2.0     994
    1.0     766
    Name: rating, dtype: int64



### 2.7. Save 3-class data to a pickle file `df_reviews_3classes.pkl`

```python
with open('df_reviews_3classes.pkl', 'wb') as f:
    pickle.dump(df, f)
```

## 3. Topic Modeling using Latent Dirichlet Allocation (LDA)

Topic modeling is among the common uses of Natural Language Processing (NLP) for extracting main ideas from multiple documents. There are quite a few ways to do this but for purposes of this study, we will use LDA. LDA is a generative probabilistic model that assumes each topic is a combination over a set of words, and each document is a mixture of over a set of topic probabilities.

The way LDA does topic modeling is that each topic is simply a collection of words (the topics) in a certain proportion. For example, `topic 0` will be represented as `(Word1*0.007, Word2*0.003, Word3*0.0012, ... Wordn*xxxx)`, where the top most likely words in the topic are displayed alongside their respective probability of occurring. These probabilities of course but sum to `1`. 

#### Doing LDA in Python

For this study, we use the LDA model available through the `gensim` library in Python, which is home to some more recent NLP algorithms. These methods will be discussed later in this section of the notebook.

### 3.1. Load original data


```python
pickle_in = open("DF_glassdoor_3718.pkl","rb")
df = pickle.load(pickle_in)
```

### 3.2. Define function for LDA


```python
def lda_prep(docs):
    """
    Creates a document-term matrix for LDA application.
    
    Parameters
    ----------
    docs : list of strings
    
    Returns
    -------
    doc_term_matrix : array
        Use this as input for the LDA model
    dictionary : something
        this one too
    """
    docs2 = [x.split(' ') for x in docs]
    dictionary = corpora.Dictionary(docs2)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs2]
    
    return doc_term_matrix, dictionary
```

### 3.3. Create corpus from reviews


```python
corp = list(df['review'])
```


```python
dt_matrix, mdict = lda_prep(corp)
```

### 3.4. Creating the object for LDA model using gensim library


```python
Lda = gensim.models.ldamodel.LdaModel
```

### 3.5. Running and training LDA model on the document term matrix


```python
ldamodel = Lda(dt_matrix, num_topics=4, id2word = mdict, passes=50, random_state=42)
```


```python
ldamodel.print_topics(num_words=10)
```




    [(0,
      '0.028*"employee" + 0.012*"management" + 0.012*"benefit" + 0.011*"pay" + 0.010*"company" + 0.009*"salary" + 0.008*"people" + 0.008*"project" + 0.007*"time" + 0.007*"lot"'),
     (1,
      '0.015*"people" + 0.015*"company" + 0.015*"employee" + 0.011*"time" + 0.008*"management" + 0.007*"great" + 0.006*"hour" + 0.006*"salary" + 0.006*"go" + 0.006*"day"'),
     (2,
      '0.026*"company" + 0.015*"people" + 0.014*"employee" + 0.009*"pay" + 0.009*"management" + 0.008*"benefit" + 0.008*"training" + 0.007*"salary" + 0.006*"great" + 0.006*"job"'),
     (3,
      '0.025*"salary" + 0.021*"employee" + 0.017*"benefit" + 0.013*"low" + 0.012*"company" + 0.010*"give" + 0.009*"high" + 0.008*"people" + 0.008*"environment" + 0.008*"management"')]



Below is an implementation of pyLDAvis, which visualizes LDA results.


```python
out5 = pyLDAvis.gensim.prepare(ldamodel, dt_matrix, mdict)
out5
```

<img src="https://github.com/jeddahlyngacera/labor-glassdoor/blob/master/out5.PNG">

### 3.6. Turning the probabilities into features

It is possible to turn the output of the LDA model into features since its assignment of an input string to a topic is probability based: that is, it outputs the probabilities of how likely that string belongs to each topic. In the case of `n=4`, the output per string is an array of length 4, with probabilities per topic. 

We create the function `get_topic` and use it to produce our features in a new a dataframe.


```python
def get_topic(input_string):
    new_str = input_string.split(' ')
    new_doc_bow = mdict.doc2bow(new_str)
    probs = ldamodel.get_document_topics(new_doc_bow)
    probs.sort(key=lambda x: x[0])
    l = [x[1] for x in probs]
    return l

df['topic_cluster'] = df['review'].apply(lambda x: get_topic(x))
df_probs = pd.DataFrame(df['topic_cluster'].values.tolist(), columns=['p1','p2','p3','p4']).fillna(0)
df_probs['rating']=df['rating']
```


```python
df_probs.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p1</th>
      <th>p2</th>
      <th>p3</th>
      <th>p4</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3713</th>
      <td>0.447661</td>
      <td>0.540785</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3714</th>
      <td>0.316928</td>
      <td>0.558146</td>
      <td>0.116279</td>
      <td>0.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3715</th>
      <td>0.024281</td>
      <td>0.506820</td>
      <td>0.023928</td>
      <td>0.444970</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3716</th>
      <td>0.955028</td>
      <td>0.014777</td>
      <td>0.015083</td>
      <td>0.015112</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3717</th>
      <td>0.029256</td>
      <td>0.029174</td>
      <td>0.400387</td>
      <td>0.541183</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
with open("df_probs_4.pkl", "wb") as f:
    pickle.dump(df_probs, f)
```

## 4. Deep Learning Classification Model

Word embedding is a method in Natural Language Processing (NLP) wherein text data is then transformed into vector representations. With this, the vector representation of each text is then able to capture the semantic and syntactic context of each word relative to the corpus. This simply means that words of similar meanings would be represented with highly similar vectors (Lai, S., et al.,2016). Vectorizing the text data enables its use for further analysis such as machine and statistical analysis with consideration of its context relative to a corpus. 

In the study, word embedding was used to initialize the text data prior to feeding it into the deep learning classification model. This was executed by incorporating a Tensorflow Embedding layer into the deep learning model pipeline. The Embedding layer serves as a lookup table which maps the words based on its indices to dense vectors, correspondingly its embedding. With this, the model was able to vectorize the text data of each company review through word embedding.

<img src="https://github.com/jeddahlyngacera/labor-glassdoor/blob/master/img2.PNG">
<p style="text-align: center;">Figure 2. Comparison of LSTM and GRU architecture</p>

In classifying the sentiments, as represented by the star ratings in each company review, a stacked Gated Recurrent Unit (GRU) model was used alongside Dense layers and an Activation layer. The GRU layer follows the Recurrent Neural Network (RNN) architecture and is closely comparable to Long Short Term Memory (LSTM) – Figure 1 illustrates the difference between LSTM and GRU models. Moreover, the distinction of GRU models are its reset and update gate. In particular, the model determines how it would integrate previous memory with the new input through the reset gate whereas the update gate determines by how much of the previous memory is to be retained by the model. 

Given the aforementioned deep learning architecture, a comparison was made between drawing a sentiment analysis by incorporating topic modelling into the deep learning model and directly implementing a deep learning model without topic modelling to execute a sentiment analysis. The first deep learning model merely uses the text data as a corpus without prior classification or clustering whereas the comparative model clusters the corpus into different topics through LDA and uses this clustering to execute a deep learning sentiment analysis per topic cluster.

Additionally, in evaluating the model, the model accuracy, precision, and recall, were used in verifying the model’s overall performance in running a sentiment analysis. This value was derived using Tensorflow’s Metric library. The accuracy accounts the ratio between the number of properly classified items relative to the overall count of predicted values. This was further evaluated relative to the Proportion Chance Criterion (PCC) of the data.

## Deep Learning Classification Model (without topic modeling)
### 4.1. Load 3-class data


```python
with open('df_reviews_3classes.pkl', 'rb') as f:
    df = pickle.load(f)
```


```python
df.rating.unique()
```




    array(['2.0', '3.0', '1.0'], dtype=object)



### 4.2. Balance data by undersampling


```python
df1 = df[df.rating=='1.0']
df2 = df[df.rating=='2.0']
df3 = df[df.rating=='3.0']
```


```python
least_count = df.rating.value_counts().min()
df1 = df1.sample(least_count)
df2 = df2.sample(least_count)
df3 = df3.sample(least_count)
```


```python
df = df1.append(df2).append(df3).reset_index(drop=True)
df.shape
```




    (2298, 2)



### 4.3. Compute baseline (`1.25*PCC`)


```python
def pcc(y):
    tc = np.unique(y, return_counts=True)[1]
    pcc = np.sum((tc/tc.sum())**2)
    return pcc
```


```python
print('PCC =', pcc(df.rating))
print('1.25*PCC =', 1.25*pcc(df.rating))
```

    PCC = 0.3333333333333333
    1.25*PCC = 0.41666666666666663
    

### 4.4. Convert text to features using `Tokenizer`


```python
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(list(df['review']))
X = tokenizer.texts_to_sequences(list(df['review']))
X = pad_sequences(X)
Y = pd.get_dummies(df['rating']).values
```

### 4.5. Split data using `train_test_split`


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, 
                                                    random_state = 42)
print('train data:', X_train.shape, Y_train.shape)
print('test data:', X_test.shape, Y_test.shape)
```

    train data: (1608, 357) (1608, 3)
    test data: (690, 357) (690, 3)
    

### 4.6. Define model callbacks


```python
# checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
```

### 4.7. Define NN model layers


```python
model = Sequential()
model.add(Embedding(300, 100, input_length = X.shape[1]))
model.add(GRU(256))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='tanh'))
model.add(Activation('softmax'))
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 357, 100)          30000     
    _________________________________________________________________
    gru (GRU)                    (None, 256)               274944    
    _________________________________________________________________
    dense (Dense)                (None, 100)               25700     
    _________________________________________________________________
    dense_1 (Dense)              (None, 3)                 303       
    _________________________________________________________________
    activation (Activation)      (None, 3)                 0         
    =================================================================
    Total params: 330,947
    Trainable params: 330,947
    Non-trainable params: 0
    _________________________________________________________________
    

### 4.8. Fit model


```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Note that for simplicity, only 3 epochs are shown below:


```python
model.fit(X_train, Y_train, epochs=3, batch_size=32, verbose=1, 
          callbacks=[checkpoint, lr_reduce], validation_data=(X_test, Y_test));
```

    Train on 1608 samples, validate on 690 samples
    Epoch 1/3
    1608/1608 [==============================] - 8s 5ms/sample - loss: 0.6245 - accuracy: 0.6716 - val_loss: 0.5782 - val_accuracy: 0.6995
    Epoch 2/3
    1608/1608 [==============================] - 4s 2ms/sample - loss: 0.5639 - accuracy: 0.7102 - val_loss: 0.5652 - val_accuracy: 0.7106
    Epoch 3/3
    1608/1608 [==============================] - 4s 2ms/sample - loss: 0.6512 - accuracy: 0.6824 - val_loss: 0.9031 - val_accuracy: 0.5643
    

### 4.9. Load model with best weights

After modeling (fitting is incremental so code above was just rerun to improve model), we load the best weight saved to the file defined in the callbacks.


```python
model.load_weights('best_weights.hdf5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### 4.10. Predict classification on `X_test`


```python
preds = model.predict(X_test)
pred_l = np.zeros_like(preds)
pred_l[np.arange(len(preds)), preds.argmax(1)] = 1
```

### 4.11. Compute accuracy of the model


```python
a = Accuracy()
a.update_state(Y_test, pred_l)
print('Accuracy:', a.result().numpy())

p = Precision()
p.update_state(Y_test, pred_l)
print('Precision:', p.result().numpy())

r = Recall()
r.update_state(Y_test, pred_l)
print('Recall:', r.result().numpy())
```

    Accuracy: 0.668599
    Precision: 0.5028986
    Recall: 0.5028986
    

## 5. Deep Learning Classification Model (with topic modeling)
### Classification of ratings per LDA cluster

### 5.1. Load 3-class data


```python
with open('df_reviews_3classes.pkl', 'rb') as f:
    df = pickle.load(f)
```


```python
df.rating.unique()
```




    array(['2.0', '3.0', '1.0'], dtype=object)



### 5.2. Load LDA result


```python
with open('df_probs_4.pkl', 'rb') as f:
    df2 = pickle.load(f)
```


```python
df2.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p1</th>
      <th>p2</th>
      <th>p3</th>
      <th>p4</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.031846</td>
      <td>0.341236</td>
      <td>0.347344</td>
      <td>0.279573</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.013853</td>
      <td>0.548061</td>
      <td>0.014329</td>
      <td>0.423757</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.377836</td>
      <td>0.017038</td>
      <td>0.510409</td>
      <td>0.094717</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.010039</td>
      <td>0.212968</td>
      <td>0.139367</td>
      <td>0.637626</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.029290</td>
      <td>0.441617</td>
      <td>0.029088</td>
      <td>0.500005</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2['probs'] = df2[['p1', 'p2', 'p3', 'p4']].values.tolist()
df2['topic'] = df2['probs'].apply(lambda x: np.argmax(x))
```

### 5.3. Combine the 2 dataframes


```python
df = df.join(df2['topic'])
```


```python
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>rating</th>
      <th>topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>accomodate staff clean environment think con c...</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>1</td>
      <td>well know company line money remittance pawn j...</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>know country big company pawnshop great benefi...</td>
      <td>3.0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>acra division family life always render overti...</td>
      <td>2.0</td>
      <td>3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>salary always time much pressure especially se...</td>
      <td>2.0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### 5.4. Split into clusters


```python
df_0 = df[df.topic==0]
df_1 = df[df.topic==1]
df_2 = df[df.topic==2]
df_3 = df[df.topic==3]
```

### 5.4. A. `Cluster 0`


```python
df_0.rating.value_counts()
```




    3.0    423
    1.0    279
    2.0    213
    Name: rating, dtype: int64




```python
def pcc(y):
    tc = np.unique(y, return_counts=True)[1]
    pcc = np.sum((tc/tc.sum())**2)
    return pcc
```


```python
print('PCC =', pcc(df_0.rating))
print('1.25*PCC =', 1.25*pcc(df_0.rating))
```

    PCC = 0.3608814834721849
    1.25*PCC = 0.45110185434023115
    

### A.1. Convert text to features using `Tokenizer`


```python
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(list(df_0['review']))
X = tokenizer.texts_to_sequences(list(df_0['review']))
X = pad_sequences(X)
Y = pd.get_dummies(df_0['rating']).values
```

### A.2. Split data using `train_test_split`


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, 
                                                    random_state = 42)
print('train data:', X_train.shape, Y_train.shape)
print('test data:', X_test.shape, Y_test.shape)
```

    train data: (640, 353) (640, 3)
    test data: (275, 353) (275, 3)
    

### A.3. Define model callbacks


```python
# checkpoint = ModelCheckpoint(filepath='best_weights_0.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
```

### A.4. Define NN model layers


```python
model_0 = Sequential()
model_0.add(Embedding(300, 100, input_length = X.shape[1]))
model_0.add(GRU(256))
model_0.add(Dense(100, activation='relu'))
model_0.add(Dense(3, activation='tanh'))
model_0.add(Activation('softmax'))
model_0.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 353, 100)          30000     
    _________________________________________________________________
    gru_1 (GRU)                  (None, 256)               274944    
    _________________________________________________________________
    dense_2 (Dense)              (None, 100)               25700     
    _________________________________________________________________
    dense_3 (Dense)              (None, 3)                 303       
    _________________________________________________________________
    activation_1 (Activation)    (None, 3)                 0         
    =================================================================
    Total params: 330,947
    Trainable params: 330,947
    Non-trainable params: 0
    _________________________________________________________________
    

### A.5. Fit model


```python
model_0.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Note that for simplicity, only 3 epochs are shown below:


```python
model_0.fit(X_train, Y_train, epochs=3, batch_size=32, verbose=1, 
          callbacks=[checkpoint, lr_reduce], validation_data=(X_test, Y_test));
```

    Train on 640 samples, validate on 275 samples
    Epoch 1/3
    640/640 [==============================] - 5s 8ms/sample - loss: 0.6183 - accuracy: 0.6719 - val_loss: 0.6016 - val_accuracy: 0.6642
    Epoch 2/3
    640/640 [==============================] - 2s 2ms/sample - loss: 0.5608 - accuracy: 0.7094 - val_loss: 0.5734 - val_accuracy: 0.7030
    Epoch 3/3
    576/640 [==========================>...] - ETA: 0s - loss: 0.5101 - accuracy: 0.7668
    Epoch 00003: ReduceLROnPlateau reducing learning rate to 0.0003000000142492354.
    640/640 [==============================] - 2s 3ms/sample - loss: 0.5030 - accuracy: 0.7724 - val_loss: 0.5537 - val_accuracy: 0.7212
    

### A.6. Load model with best weights

After modeling (fitting is incremental so code above was just rerun to improve model), we load the best weight saved to the file defined in the callbacks.


```python
model_0.load_weights('best_weights_0.hdf5')
model_0.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### A.7. Predict classification on `X_test`


```python
preds = model_0.predict(X_test)
pred_l = np.zeros_like(preds)
pred_l[np.arange(len(preds)), preds.argmax(1)] = 1
```

### A.8. Compute accuracy of the model


```python
a = Accuracy()
a.update_state(Y_test, pred_l)
print('Accuracy:', a.result().numpy())

p = Precision()
p.update_state(Y_test, pred_l)
print('Precision:', p.result().numpy())

r = Recall()
r.update_state(Y_test, pred_l)
print('Recall:', r.result().numpy())
```

    Accuracy: 0.7309091
    Precision: 0.59636366
    Recall: 0.59636366
    

### 5.4. B. `Cluster 1`


```python
df_1.rating.value_counts()
```




    3.0    551
    1.0    246
    2.0    235
    Name: rating, dtype: int64




```python
def pcc(y):
    tc = np.unique(y, return_counts=True)[1]
    pcc = np.sum((tc/tc.sum())**2)
    return pcc
```


```python
print('PCC =', pcc(df_1.rating))
print('1.25*PCC =', 1.25*pcc(df_1.rating))
```

    PCC = 0.39373948380505974
    1.25*PCC = 0.49217435475632465
    

### B.1. Convert text to features using `Tokenizer`


```python
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(list(df_1['review']))
X = tokenizer.texts_to_sequences(list(df_1['review']))
X = pad_sequences(X)
Y = pd.get_dummies(df_1['rating']).values
```

### B.2. Split data using `train_test_split`


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, 
                                                    random_state = 42)
print('train data:', X_train.shape, Y_train.shape)
print('test data:', X_test.shape, Y_test.shape)
```

    train data: (722, 209) (722, 3)
    test data: (310, 209) (310, 3)
    

### B.3. Define model callbacks


```python
# checkpoint = ModelCheckpoint(filepath='best_weights_1.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
```

### B.4. Define NN model layers


```python
model_1 = Sequential()
model_1.add(Embedding(300, 100, input_length = X.shape[1]))
model_1.add(GRU(256))
model_1.add(Dense(100, activation='relu'))
model_1.add(Dense(3, activation='tanh'))
model_1.add(Activation('softmax'))
model_1.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 209, 100)          30000     
    _________________________________________________________________
    gru_2 (GRU)                  (None, 256)               274944    
    _________________________________________________________________
    dense_4 (Dense)              (None, 100)               25700     
    _________________________________________________________________
    dense_5 (Dense)              (None, 3)                 303       
    _________________________________________________________________
    activation_2 (Activation)    (None, 3)                 0         
    =================================================================
    Total params: 330,947
    Trainable params: 330,947
    Non-trainable params: 0
    _________________________________________________________________
    

### B.5. Fit model


```python
model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Note that for simplicity, only 3 epochs are shown below:


```python
model_1.fit(X_train, Y_train, epochs=3, batch_size=32, verbose=1, 
          callbacks=[checkpoint, lr_reduce], validation_data=(X_test, Y_test));
```

    Train on 722 samples, validate on 310 samples
    Epoch 1/3
    722/722 [==============================] - 4s 5ms/sample - loss: 0.6001 - accuracy: 0.6814 - val_loss: 0.5904 - val_accuracy: 0.6839
    Epoch 2/3
    722/722 [==============================] - 1s 2ms/sample - loss: 0.5607 - accuracy: 0.7322 - val_loss: 0.5559 - val_accuracy: 0.7333
    Epoch 3/3
    704/722 [============================>.] - ETA: 0s - loss: 0.4986 - accuracy: 0.7760
    Epoch 00003: ReduceLROnPlateau reducing learning rate to 0.0003000000142492354.
    722/722 [==============================] - 1s 2ms/sample - loss: 0.4984 - accuracy: 0.7761 - val_loss: 0.5641 - val_accuracy: 0.7183
    

### B.6. Load model with best weights

After modeling (fitting is incremental so code above was just rerun to improve model), we load the best weight saved to the file defined in the callbacks.


```python
model_1.load_weights('best_weights_1.hdf5')
model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### B.7. Predict classification on `X_test`


```python
preds = model_1.predict(X_test)
pred_l = np.zeros_like(preds)
pred_l[np.arange(len(preds)), preds.argmax(1)] = 1
```

### B.8. Compute accuracy of the model


```python
a = Accuracy()
a.update_state(Y_test, pred_l)
print('Accuracy:', a.result().numpy())

p = Precision()
p.update_state(Y_test, pred_l)
print('Precision:', p.result().numpy())

r = Recall()
r.update_state(Y_test, pred_l)
print('Recall:', r.result().numpy())
```

    Accuracy: 0.73333335
    Precision: 0.6
    Recall: 0.6
    

### 5.4. C. `Cluster 2`


```python
df_2.rating.value_counts()
```




    3.0    425
    2.0    214
    1.0    100
    Name: rating, dtype: int64




```python
def pcc(y):
    tc = np.unique(y, return_counts=True)[1]
    pcc = np.sum((tc/tc.sum())**2)
    return pcc
```


```python
print('PCC =', pcc(df_2.rating))
print('1.25*PCC =', 1.25*pcc(df_2.rating))
```

    PCC = 0.43290955667333797
    1.25*PCC = 0.5411369458416725
    

### C.1. Convert text to features using `Tokenizer`


```python
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(list(df_2['review']))
X = tokenizer.texts_to_sequences(list(df_2['review']))
X = pad_sequences(X)
Y = pd.get_dummies(df_2['rating']).values
```

### C.2. Split data using `train_test_split`


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, 
                                                    random_state = 42)
print('train data:', X_train.shape, Y_train.shape)
print('test data:', X_test.shape, Y_test.shape)
```

    train data: (517, 80) (517, 3)
    test data: (222, 80) (222, 3)
    

### C.3. Define model callbacks


```python
# checkpoint = ModelCheckpoint(filepath='best_weights_2.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
```

### C.4. Define NN model layers


```python
model_2 = Sequential()
model_2.add(Embedding(300, 100, input_length = X.shape[1]))
model_2.add(GRU(256))
model_2.add(Dense(100, activation='relu'))
model_2.add(Dense(3, activation='tanh'))
model_2.add(Activation('softmax'))
model_2.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 80, 100)           30000     
    _________________________________________________________________
    gru_3 (GRU)                  (None, 256)               274944    
    _________________________________________________________________
    dense_6 (Dense)              (None, 100)               25700     
    _________________________________________________________________
    dense_7 (Dense)              (None, 3)                 303       
    _________________________________________________________________
    activation_3 (Activation)    (None, 3)                 0         
    =================================================================
    Total params: 330,947
    Trainable params: 330,947
    Non-trainable params: 0
    _________________________________________________________________
    

### C.5. Fit model


```python
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Note that for simplicity, only 3 epochs are shown below:


```python
model_2.fit(X_train, Y_train, epochs=3, batch_size=32, verbose=1, 
          callbacks=[checkpoint, lr_reduce], validation_data=(X_test, Y_test));
```

    Train on 517 samples, validate on 222 samples
    Epoch 1/3
    517/517 [==============================] - 3s 6ms/sample - loss: 0.5877 - accuracy: 0.6925 - val_loss: 0.5678 - val_accuracy: 0.7177
    Epoch 2/3
    517/517 [==============================] - 1s 2ms/sample - loss: 0.5797 - accuracy: 0.7163 - val_loss: 0.5798 - val_accuracy: 0.7177
    Epoch 3/3
    517/517 [==============================] - 1s 2ms/sample - loss: 0.5720 - accuracy: 0.7163 - val_loss: 0.5556 - val_accuracy: 0.7177
    

### C.6. Load model with best weights

After modeling (fitting is incremental so code above was just rerun to improve model), we load the best weight saved to the file defined in the callbacks.


```python
model_2.load_weights('best_weights_2.hdf5')
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### C.7. Predict classification on `X_test`


```python
preds = model_2.predict(X_test)
pred_l = np.zeros_like(preds)
pred_l[np.arange(len(preds)), preds.argmax(1)] = 1
```

### C.8. Compute accuracy of the model


```python
a = Accuracy()
a.update_state(Y_test, pred_l)
print('Accuracy:', a.result().numpy())

p = Precision()
p.update_state(Y_test, pred_l)
print('Precision:', p.result().numpy())

r = Recall()
r.update_state(Y_test, pred_l)
print('Recall:', r.result().numpy())
```

    Accuracy: 0.7237237
    Precision: 0.5855856
    Recall: 0.5855856
    

### 5.4. D. `Cluster 3`


```python
df_3.rating.value_counts()
```




    3.0    559
    2.0    332
    1.0    141
    Name: rating, dtype: int64




```python
def pcc(y):
    tc = np.unique(y, return_counts=True)[1]
    pcc = np.sum((tc/tc.sum())**2)
    return pcc
```


```python
print('PCC =', pcc(df_3.rating))
print('1.25*PCC =', 1.25*pcc(df_3.rating))
```

    PCC = 0.4155643440898984
    1.25*PCC = 0.519455430112373
    

### D.1. Convert text to features using `Tokenizer`


```python
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(list(df_3['review']))
X = tokenizer.texts_to_sequences(list(df_3['review']))
X = pad_sequences(X)
Y = pd.get_dummies(df_3['rating']).values
```

### D.2. Split data using `train_test_split`


```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, 
                                                    random_state = 42)
print('train data:', X_train.shape, Y_train.shape)
print('test data:', X_test.shape, Y_test.shape)
```

    train data: (722, 140) (722, 3)
    test data: (310, 140) (310, 3)
    

### D.3. Define model callbacks


```python
# checkpoint = ModelCheckpoint(filepath='best_weights_3.hdf5', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
```

### D.4. Define NN model layers


```python
model_3 = Sequential()
model_3.add(Embedding(300, 100, input_length = X.shape[1]))
model_3.add(GRU(256))
model_3.add(Dense(100, activation='relu'))
model_3.add(Dense(3, activation='tanh'))
model_3.add(Activation('softmax'))
model_3.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_4 (Embedding)      (None, 140, 100)          30000     
    _________________________________________________________________
    gru_4 (GRU)                  (None, 256)               274944    
    _________________________________________________________________
    dense_8 (Dense)              (None, 100)               25700     
    _________________________________________________________________
    dense_9 (Dense)              (None, 3)                 303       
    _________________________________________________________________
    activation_4 (Activation)    (None, 3)                 0         
    =================================================================
    Total params: 330,947
    Trainable params: 330,947
    Non-trainable params: 0
    _________________________________________________________________
    

### D.5. Fit model


```python
model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Note that for simplicity, only 3 epochs are shown below:


```python
model_3.fit(X_train, Y_train, epochs=3, batch_size=32, verbose=1, 
          callbacks=[checkpoint, lr_reduce], validation_data=(X_test, Y_test));
```

    Train on 722 samples, validate on 310 samples
    Epoch 1/3
    722/722 [==============================] - 4s 5ms/sample - loss: 0.5922 - accuracy: 0.6833 - val_loss: 0.6136 - val_accuracy: 0.6667
    Epoch 2/3
    722/722 [==============================] - 1s 2ms/sample - loss: 0.5965 - accuracy: 0.6667 - val_loss: 0.6148 - val_accuracy: 0.6667
    Epoch 3/3
    722/722 [==============================] - 1s 2ms/sample - loss: 0.5921 - accuracy: 0.6727 - val_loss: 0.5747 - val_accuracy: 0.7054
    

### D.6. Load model with best weights

After modeling (fitting is incremental so code above was just rerun to improve model), we load the best weight saved to the file defined in the callbacks.


```python
model_3.load_weights('best_weights_3.hdf5')
model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### D.7. Predict classification on `X_test`


```python
preds = model_3.predict(X_test)
pred_l = np.zeros_like(preds)
pred_l[np.arange(len(preds)), preds.argmax(1)] = 1
```

### D.8. Compute accuracy of the model


```python
a = Accuracy()
a.update_state(Y_test, pred_l)
print('Accuracy:', a.result().numpy())

p = Precision()
p.update_state(Y_test, pred_l)
print('Precision:', p.result().numpy())

r = Recall()
r.update_state(Y_test, pred_l)
print('Recall:', r.result().numpy())
```

    Accuracy: 0.70107526
    Precision: 0.5516129
    Recall: 0.5516129
    

# Results and Discussion
<img src="https://github.com/jeddahlyngacera/labor-glassdoor/blob/master/img3.PNG">
<p style="text-align: center;">Figure 3. Topic distribution through LDA on company reviews</p>

By running LDA, it has been found that the four distinct clusters are company culture, career growth, renumeration, and management respectively. Furthermore, as seen in Figure 3., it has been found that 35.7% of the company reviews talk about company culture; career growth, renumeration, and management follows with 24.1%, 23.1%, and 17.2% respectively. With this, the results of topic modeling strongly shows that that career growth occurs at 35.7% of the company reviews analyzed and the following topic clusters follow the same interpretation relative to their topic distribution. These topic clusters pertain to the following concerns about the company:
* **Topic Cluster 0: Company culture**
    - The inter and intra- personal dynamics within the company
* **Topic Cluster 1: Career growth**
    - Pertains to trainings, accreditations, and recognition in and out of the company
* **Topic Cluster 2: Renumeration**
    - Pertains to monetary and other benefits given by the company this includes allowances, health insurance, and salary
* **Topic Cluster 3: Management**
    - Pertains to the overall management on the employees, which encompasses time management, load assignment, and management decisions.

<img src="https://github.com/jeddahlyngacera/labor-glassdoor/blob/master/img4.PNG">

As seen in the table above, it is apparent that higher accuracies may be derived when a combination of both topic modelling and deep learning models is implemented. Nonetheless, without topic modelling, straightforward GRU model was still able to surpass the benchmark set by the 1.25 PCC by yielding an accuracy of 66.86%. However, taking into account the average accuracy for each of the topic clusters in the hybrid model, it is seen that this has yielded a higher benchmark as given by the 1.25 PCC of 50.10% and accordingly, the models yielded a higher average accuracy of 72.23%. 

Different baselines were imposed for each topic cluster in the hybrid model due to the imbalance of sentiments within each topic cluster. Nonetheless, by setting the straightforward model without topic modelling as a baseline, shows that each of the topic clusters were able to surpass this set baseline on all accounts of accuracy, precision, and recall. 
However, a distinct observation may be drawn on the precision and recall draw from each of the models. It is seen that each of the precision and recall for each model have the same values and drawing from the definition of precision and recall, this implies that the false positives and false negatives in the classifications were equal. This further suggests that the models were able to balance out the false negatives and false positives across different classes through its algorithms.

# Conclusion/Recommendation

From the results we gathered, we see that utilizing a Stacked GRU model and word embeddings outperforms our baseline model with a 67 % accuracy, with precision and recall scores of both over 50%. Our unique methodology that incorporates topic modeling in our pipeline increased the accuracy up to 73%, with precision and recall scores of 60%, a 10% increase from our original Stacked GRU. This shows that our hybrid model of clustering and deep neural networks is effective in predicting sentiments-based outcomes on textual data.

It is worth mentioning that since our model also shows the general reasons that contribute to the sentiment of the review, it is more useful to companies compared to merely providing predictions on the basis of textual review. This added functionality could offer a better understanding of the dynamics affecting employer-employee relations. 

One of the limitations faced by our study is in the gathering of a larger Philippines-based data set. Compared to other labor markets, the Philippines based companies enrolled in Glassdoor.com is quite small. A more robust dataset would have provided our algorithm with better training data that could improve the prediction performance of our models. Neural networks perform best when trained with larger datasets (Rolnick, 2017). Another limitation is language. The Philippines is a bi-lingual culture, the common vernacular is a mixture of English and Filipino. Some reviews in the website are a mixture of both English and Filipino words. A more inclusive model incorporating Filipino words in the predictions would be ideal.

In our study, we have shown that our models could accurately predict the quantitative sentiment of an employee (positive, negative or neutral) from qualitative comments. Although something similar has been done before, i.e. in the prediction of product reviews in Amazon, our approach gives further insights on what reasons are the basis for such a rating. Our Stacked GRU model (without any clustering) could effectively predict sentiment with a 67% accuracy. This can further be improved to up to 73% once clustering of the reasons for giving the rating have been employed. Being able to identify specific topics that contribute to positive or negative ratings could potentially offer managers and other policy-makers in the company deeper insights on how to improve employer-employee relations in the future. Actions that could translate to cost-savings and improve the bottom line of the company. 

Additional applications of the study include being able to measure employee engagement from internal company feedback mechanisms such as “the voice of the employee” in real time which could provide employers a quick “feel” of the overall morale of the workforce. Further studies could also be done on how to improve accuracies on a 5-category classification model to provide a more detailed prediction of popular rating systems.


# Acknowledgement

Special thanks to Professor Christopher Monterola, Professor Erika Legara, and Associate Professor Eduardo David of the Asian Institute of Management (Manila, Philippines) for their guidance and support. 


```python

```
