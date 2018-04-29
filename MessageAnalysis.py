
# coding: utf-8

# In[80]:


from pymongo import MongoClient
import pickle
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import text_to_word_sequence
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Bidirectional

FILTERS = """''!"#$%&()*+,-./:;<=>?@[\]^_`{|}~    
।,!?+০১২৩৪৫৬৭৮৯0123456789'
'\u3000',
 '亚',
 '法',
 '瑞',
 '민',
 '샤',
\x08\t\r\xad'à',
 'á',
 'â',
 'ã',
 'ä',
 'å',
 'ç',
 'è',
 'é',
 'ê',
 'ë',
 'ì',
 'í',
 'î',
 'ï',
 'ñ',
 'ò',
 'ô',
 'ö',
 'ø',
 'ü',
 'ý',
 'ÿ',
 'ā',
 'ă',
 'ą',
 'ď',
 'ē',
 'ę',
 'ě',
 'ń',
 'ņ',
 'ő',
 'ŕ',
 'ř',
 'ś',
 'ş',
 'š',
 'ţ',
 'ū',
 '̶',
 'أ',
 'ئ',
 'ا',
 'ة',
 'ج',
 'ر',
 'ز',
 'س',
 'ش',
 'ط',
 'ع',
 'ف',
 'ق',
 'ل',
 'م',
 'ن',
 'ه',
 'و',
 'ي',
'\u200b',
 '\u200c',
 '\u200d',
'•',
 '…',
 '\u202c',
 '\u202d',
 '₪',
 '↗',
 '●',
 '★',
 '☆',
 '☺',
 '♥',
 '♦',
 '⚠',
 '✅',
 '✔',
 '✴',
 '✿',
 '❤',
 '➡',
 '\u3000',
 '亚',
 '法',
 '瑞',
 '민',
 '샤',
 '️',
 'ﻋ',
 'ﻥ',
 'ﻮ',
 'ﻴ',
 'ｂ',
 '�',
 '🇦',
 '🇸',
 '🇺',
 '🏻',
 '🏼',
 '👉',
 '👍',
 '💐',
 '💓',
 '💘',
 '💙',
 '💜',
 '💟',
 '🔴',
 '🔸',
 '🔺',
 '৷',
 '৺',
'–',
 '—',
 '’',
 '‚',
 '“',
 '”'
"""

HOST = 'localhost'
PORT = 27017

mongo = MongoClient(HOST, PORT)
db = mongo['shopup']
collection = db['fbPageConversation']

NUM_CLASS = 7


# labels 
LABELS = {
    'orderConfirmed' : 0, 
    'orderCancelled' : 1, 
    'productPrice' : 2,
    'deliveryDate' : 3,
    'available' : 4,
    'query' : 5,
    'outOfStock' : 6,
    'others' : 7
}

def label_to_index(input_dict, one_hot=False):
    try:
        for key in input_dict:
            if input_dict[key]:
                if one_hot == True:
                    return np.eye(NUM_CLASS)[LABELS[key]]
                else:
                    return LABELS[key]
    except:
        return None
        
        
def get_messages_labels_by_thread(thread):
    messages = []
    labels = []
    
    for message in thread['messages']['data']:
        try:
            y = message['labels']
            labels.append( label_to_index( y ) )
            messages.append(message['message'])
        except:
            pass
    return (messages, labels)


# In[113]:


if 1 == 0:
    message_data_iter = collection.find( {"$and" : [{'status' : "done"}, {'messages.data.labels.outOfStock' :  { "$exists" : True} }] })


# In[114]:


with open('shopup_labeled_thread.pkl', 'rb') as f:
    data = pickle.load(f)


# In[115]:


if 1 == 0:
    with open('shopup_labeled_thread.pkl', 'wb') as f:
        pickle.dump(all_datas, f)


# # Filtering Valid Message With Labels 

# In[292]:


if 1 == 0:
    messages = []
    labels = []
    valid_indices = []

    for thread in tqdm(data):
        try:
            _messages, _labels = get_messages_labels_by_thread(thread)

            for msg, label in zip(_messages, _labels):
                messages.append(msg)
                labels.append(label)
        except:
            pass


    for i, l in enumerate(labels):
        if np.any(l) != None:
            valid_indices.append(i)


    valid_messages = np.array(messages)[valid_indices].tolist()
    valid_labels = np.array([ x for x in np.array(labels)[valid_indices].tolist()  ])


# In[293]:


x_tokenized = [ text_to_word_sequence(sentence, filters=FILTERS) for sentence in tqdm(valid_messages) ]


# In[302]:


if 1 == 0:
    char_set = []
    for sentence in tqdm(x_tokenized):
        for word in sentence:
            for c in word:
                char_set.append(c)
    
    with open('char_set.pkl', 'wb') as f:
        pickle.dump(set(char_set), f)
        
else:
    with open('char_set.pkl', 'rb') as f:
        char_set = pickle.load(f)


# In[329]:


if 1 == 0:
    vocabulary = Dictionary(x_tokenized)
    vocabulary.save('voca')
else:
    vocabulary = Dictionary.load('voca')
    

word_to_idx = { vocabulary[idx]:idx for idx in range(len(vocabulary)) }
idx_to_word = { idx:vocabulary[idx] for idx in range(len(vocabulary))}

char_to_idx = { c:idx for idx, c in enumerate(char_set) }
idx_to_char = { idx:c for idx, c in enumerate(char_set)}

def hf(word):
    return word_to_idx[word]


# In[83]:


train_x = pad_sequences([ char_hashing_trick(x) for x in x_tokenized  ], maxlen=200)


# In[84]:


max_features = 121

model = Sequential()
model.add(Embedding(max_features, 50))
model.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(30, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(NUM_CLASS, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(train_x, valid_labels, validation_split=0.2, shuffle=True,
          batch_size=32,
          epochs=15)


# In[85]:


model.save('max_len_200')


# In[86]:


model.fit(train_x, valid_labels, validation_split=0.2, shuffle=True,
          batch_size=32,
          epochs=15)


# In[87]:


model.save('val_acc_9488_maxlen_200.h5')

