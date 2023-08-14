from cgi import test
import tweepy
import configparser
import pandas as pd
import numpy as np
from emot.emo_unicode import UNICODE_EMOJI
import re
from nltk.tokenize import wordpunct_tokenize


#read configs
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

#authentication
auth = tweepy.OAuth1UserHandler(api_key,api_key_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

#q = success also gives us the tweets to be searched

n=1600000

tweets=tweepy.Cursor(api.search_tweets,q="peace -filter:links -filter:retweets",lang="en",tweet_mode="extended").items(n)


tweets_json=[]

for tweet in tweets:
   tweets_json.append(tweet._json)
   
x = len(tweets_json)

# To save the data as collected from twitter

import json

for i in tweets_json:
    with open("basic_data.json", "w") as outfile:
        json.dump(i, outfile)

df = pd.DataFrame(columns = ['target','ids','date','flag','user','text'])

def convert_emojis(text):
    for emot in UNICODE_EMOJI:
        text = text.replace(emot," "+UNICODE_EMOJI[emot].replace(",","").replace(":","")+" ")
        text = text.replace("_"," ")
    return text


artext = []
user_id = []
timing = []
user_name = []

for i in range(x):
    basic = tweets_json[i]["full_text"]
    basic = convert_emojis(basic)
    artext.append(basic)
    user_id.append(tweets_json[i]["id"])
    user_name.append(tweets_json[i]["user"]["name"])
    timing.append(tweets_json[i]["created_at"][:-10]+" PDT "+tweets_json[i]["created_at"][-5:])


artext = pd.Series(artext)
timing = pd.Series(timing)
user_name = pd.Series(user_name)
user_id = pd.Series(user_id)


df["text"] = artext
df["ids"] = user_id
df["date"] = timing
df["user"] = user_name
df["flag"] = "NO_QUERY"


test1=df.copy()

test1=test1.apply(lambda x:x.astype(str).str.lower())

#replace all the non-alphabetic elements with 1 and keep 
# [^A-za-z.!+ ] keep all the letter, dot and exclamation mark because a lot of of word attached to ! or . as 
# form of expression
test1["text"]=test1["text"].apply(lambda x:  re.sub("[^A-za-z.!+ ]", '1', x))

test1["text"]=test1["text"].apply(lambda x: " ".join(w for w in wordpunct_tokenize(x.strip()) if (w.isalpha())))

x = len(test1["text"])

#TextBlob api 
from textblob import TextBlob

sent = []

for i in range(x):
    blob = TextBlob(test1["text"][i])
    ss = 0
    den = 0
    for sentence in blob.sentences:
        ss += sentence.sentiment.polarity
        den += 1
    if den==0:
        sent.append(2)
    else:
        sco = ss/den
        if sco==0:
            sent.append(2)
        elif sco>0:
            sent.append(4)
        else:
            sent.append(0)

sent = pd.Series(sent,dtype="string")

test1["target"] = sent

print(test1)

test1.to_csv('testntrain.csv')