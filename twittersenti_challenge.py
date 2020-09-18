#twitter sentiment analysis
import tweepy
from textblob import TextBlob
import csv

#Authenticate
consumer_key='your own'
consumer_secret='your own'

access_token = 'your own'
access_token_secret = 'your own'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Retrieve Tweets
public_tweets=api.search('Padmavati')

##CHALLENGE - save each Tweet to a CSV file
#and label each one as either 'positive' or 'negative', depending on the sentiment 
#You can decide the sentiment polarity threshold yourself


csv_file=open('tweets.csv', 'w')
writer=csv.writer(csv_file)


for tweet in public_tweets:
	text=tweet.text.encode('utf-8').strip()
	analysis=TextBlob(tweet.text)
	polarity=analysis.sentiment.polarity

	if polarity > 0.1:
		writer.writerow([text,polarity,"positive"])

	else :
		writer.writerow([text,polarity,"negative"])

print("Results are displayed in the csv file.")	
	
