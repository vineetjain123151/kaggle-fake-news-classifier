#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[43]:


dataset=pd.read_csv('C:/Users/Veenu/Downloads/train/train.csv')


# In[44]:


dataset.head()


# In[45]:


dataset=dataset.dropna()


# In[47]:


x=dataset.drop('label',axis=1)
y=dataset['label']


# In[48]:


x


# In[49]:


x.shape


# In[50]:


y.shape


# In[51]:


import tensorflow as tf


# In[52]:


tf.__version__


# In[53]:



from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


# In[54]:


voc_size=5000


# In[55]:


messages=x.copy()


# In[56]:


messages['title'][1]


# In[59]:


messages.reset_index(inplace=True)


# In[60]:


import nltk
import re
from nltk.corpus import stopwords


# In[67]:


nltk.download('stopwords')


# In[73]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[74]:


corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['title'][i])
    review=review.lower()
    review = review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)


# In[ ]:





# In[77]:


onehot_repr=[one_hot(words,voc_size)for words in corpus]


# In[79]:


sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# In[80]:


embedded_docs[0]


# In[82]:


embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrices=['accuracy'])


# In[85]:


x_final=np.array(embedded_docs)
y_final=np.array(y)


# In[86]:


x_final.shape,y_final.shape


# In[87]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33, random_state=42)


# In[88]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


# In[94]:


y_pred=model.predict_classes(X_test)


# In[95]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[ ]:




