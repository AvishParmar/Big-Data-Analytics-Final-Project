from transformers import pipeline
import pandas as pd
import warnings 
from time import time
import random
# Print iterations progress
def printProgressBar (start, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', 'Runtime:{:0.2f}'.format(time()-start), ' | ',iteration,'/',total, end = printEnd)
    if iteration == total: 
        print()

df = pd.read_csv('./preprocessed_indexed/UkraineTweetsPreprocessed2022-03.csv',header=None)
# df = pd.read_csv('./ukraine_test.csv',header=None)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
warnings.filterwarnings('ignore')
start = time()
count = 0
# batch_size = 1024
total = len(df)
sample_size = 500000
if total > sample_size:
    random_indices = random.sample(range(total), sample_size)
    df = df.iloc[random_indices]
    total = sample_size
# total_batches = round(total/batch_size)
tweets = df.iloc[:, 2]
df[len(df.columns)] = ''
df[len(df.columns)] = ''
df.columns = ['id','date','text','hashtags','sentiment','sentiment_score']
classification_column = len(df.columns)-1
for i, tweet in tweets.iteritems():
    printProgressBar(start, count, total)
    try:
        result = classifier(str(tweet))
        count += 1
        # print(result)
        classification = result[0]['label']
        score = result[0]['score']
        df.loc[i, df.columns[-2]] = classification  # Update the 5th column explicitly
        df.loc[i, df.columns[-1]] = score
    except:
        print('Classification failed for ', i,'th row ')
        pass
# # Save the updated DataFrame to a new CSV file
print('Runtime:', time()-start)
df.to_csv('tweets_with_classification_2022_03.csv', index=False)