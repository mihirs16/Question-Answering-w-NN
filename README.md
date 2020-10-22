# Question Answering with Nearest Neighbour Search

## Downloading and Unpacking Word Embeddings
<b>Google News Vectors</b><br>
* Total Vocab: 3,000,000,000<br>
* Dimensions: 300
```bash
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gzip -d "GoogleNews-vectors-negative300.bin.gz"
```

## Importing Packages
* Pandas (DataFrame and CSV)
* Numpy (Arrays and Vector Maths)
* NLTK (Language Processing, Stopwords, WordNet)
* Gensim (Word2Vec Model)
```python
import re
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

nltk.download('wordnet')
nltk.download('stopwords')
```

## Cleaning Data
```python
df = pd.read_excel('sem-faq-db.xlsx')
df.drop([
  'Source', 
  'Metadata', 
  'SuggestedQuestions', 
  'IsContextOnly', 
  'Prompts'
 ], axis = 1, inplace = True)
df.drop([df.columns[0]], axis = 1, inplace = True)
df.head()
```

## Word2Vec Model
* Vocab Considered: 1,000,000
* Dimensions: 300
```python
en_embeddings = KeyedVectors.load_word2vec_format(
  './GoogleNews-vectors-negative300.bin', 
  binary = True, 
  limit=1000000
)
```

## Processing Data
1. Process Text
    * Remove Punctuations
    * Convert to Lowercase
    * Tokenize
    * Lemmatize all words
    * Remove Stopwords
```python
def process_que(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub("\'", "", text) 
    text = re.sub("[^a-zA-Z]"," ",text) 
    text = ' '.join(text.split()) 
    text = text.lower()
    _t = ""
    for t in text.split():
        _t += lemmatizer.lemmatize(t, pos='a') + " "
    text = _t
    stop_words = set(stopwords.words('english'))
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    text = no_stopword_text
    return text
```
2. Cosine Similarity
```python
def cosine_similarity(A, B):
    cos = -10
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)
    return cos
```
3. Vector for Each Document
```python
def get_document_embedding(que, en_embeddings): 
    doc_embedding = np.zeros(300)
    processed_doc = process_que(que)
    for word in processed_doc:
        if word in en_embeddings.vocab:        
            doc_embedding += en_embeddings[word]
        else:
            doc_embedding += 0
    return doc_embedding

exmp = df.Question.values[0]
print(exmp)
exmp_embedding = get_document_embedding(exmp, en_embeddings)
exmp_embedding[:5]
```
4. Embeddings for All Documents
```python
def get_document_vecs(all_docs, en_embeddings):
    ind2Doc_dict = {}
    document_vec_l = []
    for i, doc in enumerate(all_docs):
        doc_embedding = get_document_embedding(doc, en_embeddings)
        ind2Doc_dict[i] = doc_embedding
        document_vec_l.append(doc_embedding)
    document_vec_matrix = np.vstack(document_vec_l)
    return document_vec_matrix, ind2Doc_dict

document_vecs, ind2Tweet = get_document_vecs(df.Question.values, en_embeddings)
print(f"length of dictionary {len(ind2Tweet)}")
print(f"shape of document_vecs {document_vecs.shape}")
```

## Searching Dataset w/ Nearest Neighbour
```python
Query = input("Question: ")
que_embed = get_document_embedding(Query, en_embeddings)
idx = np.argmax(cosine_similarity(document_vecs, que_embed))
print("Matched Question: " + df.Question.values[idx]) 
print("Possible Answer: " + df.Answer.values[idx])
```
<b>Output</b><br>
![](https://github.com/mihirs16/Question-Answering-w-NN-LSH/blob/master/image.png)
