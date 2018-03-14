import keras
from keras.models import Sequential
from keras.layers import Activation,LSTM,Dense
from keras.optimizers import Adam
import pandas as pd
import numpy as np

df=pd.read_csv('sheeran.csv')['lyrics']
#df

data=np.array(df)
#print(data)

corpus=''
for ix in range(len(data)):
    corpus+=data[ix]
#corpus

vocab=list(set(corpus))
char_ix={c:i for i,c in enumerate(vocab)}
ix_char={i:c for i,c in enumerate(vocab)}
#char_ix
#ix_char

maxlen=40
vocab_size=len(vocab)

sentences=[]
next_char=[]
for i in range(len(txt)-maxlen-1):
    sentences.append(txt[i:i+maxlen])
    next_char.append(txt[i+maxlen])
#sentences
#next_char

X=np.zeros((len(sentences),maxlen,vocab_size))
y=np.zeros((len(sentences),vocab_size))
for ix in range(len(sentences)):
    y[ix,char_ix[next_char[ix]]]=1
    for iy in range(maxlen):
        X[ix,iy,char_ix[sentences[ix][iy]]]=1
#X
#y
model=Sequential()
model.add(LSTM(128,input_shape=(maxlen,vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy')

model.fit(X,y,epochs=5,batch_size=128)
#serialize model to JSON  serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


import random
generated=''
start_index=random.randint(0,len(txt)-maxlen-1)
sent=txt[start_index:start_index+maxlen]
generated+=sent
for i in range(1900):
    x_sample=generated[i:i+maxlen]
    x=np.zeros((1,maxlen,vocab_size))
    for j in range(maxlen):
        x[0,j,char_ix[x_sample[j]]]=1
    probs=model.predict(x)
    probs=np.reshape(probs,probs.shape[1])
    ix=np.random.choice(range(vocab_size),p=probs.ravel())
    generated+=ix_char[ix]
print(generated)
