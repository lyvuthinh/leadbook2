from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
import re

# # # # # # # # # 
# setting up frequency count
# # # # # # # # # 

# Load training set
job_titles = pd.read_csv('/home/thinh/workspace/leadbook2/data/training/job_titles_with_labels.csv')

# we just take a randoom of 30k training data point for faster processing, but of course, we can use the entire data set
job_titles=job_titles.sample(30000)
job_titles = job_titles.reset_index(drop=True)
job_titles["job_title"]=job_titles["job_title"].apply(lambda x: re.sub(r'[^a-zA-Z]+', ' ', x).replace("  "," ").replace("  "," ").strip())
print(job_titles["job_title"])

# set a numeric code for each department
k={'agriculture': 30,
 'art and photography': 7,
 'aviation, marine and shipping': 3,
 'business services': 13,
 'chemicals': 10,
 'clothing,cosmetics and fashion': 12,
 'construction': 11,
 'consumer services': 25,
 'customer services': 27,
 'defense': 0,
 'design': 6,
 'education': 29,
 'electrical and electronics': 8,
 'energy and mining': 17,
 'engineering and telecommunications': 31,
 'entertainment': 1,
 'financials': 33,
 'government and agencies': 9,
 'healthcare': 4,
 'hotels and culinary': 18,
 'human resources': 14,
 'import export procurement dealers and distributors': 23,
 'information technology': 21,
 'logistics and transportation': 16,
 'management': 20,
 'marketing and advertising': 15,
 'mechanical & heavy industry': 5,
 'media & journalism': 2,
 'others': 34,
 'planning and quality': 32,
 'professional services': 28,
 'real estate': 26,
 'social organisations and ngo': 24,
 'sports ,fitness ,leisure and travel': 22,
 'trade': 19}

h={0: 'defense',
 1: 'entertainment',
 2: 'media & journalism',
 3: 'aviation, marine and shipping',
 4: 'healthcare',
 5: 'mechanical & heavy industry',
 6: 'design',
 7: 'art and photography',
 8: 'electrical and electronics',
 9: 'government and agencies',
 10: 'chemicals',
 11: 'construction',
 12: 'clothing,cosmetics and fashion',
 13: 'business services',
 14: 'human resources',
 15: 'marketing and advertising',
 16: 'logistics and transportation',
 17: 'energy and mining',
 18: 'hotels and culinary',
 19: 'trade',
 20: 'management',
 21: 'information technology',
 22: 'sports ,fitness ,leisure and travel',
 23: 'import export procurement dealers and distributors',
 24: 'social organisations and ngo',
 25: 'consumer services',
 26: 'real estate',
 27: 'customer services',
 28: 'professional services',
 29: 'education',
 30: 'agriculture',
 31: 'engineering and telecommunications',
 32: 'planning and quality',
 33: 'financials',
 34: 'others'}

# convert the training data to deparment code (instead of raw job category)
y = list(
    map(
        lambda x: 
        k[x]
 , 
        job_titles.job_category
        )
    )
print(y)

# load test set
test_set = pd.read_csv('/home/thinh/workspace/leadbook2/data/jobtitles_test.csv')
# for simplicity, I just do the testing for the first 100 job titles in the test set
test_set=test_set[:100]
test_set = test_set.reset_index(drop=True)
test_set["Job Title"]=test_set["Job Title"].apply(lambda x: re.sub(r'[^a-zA-Z]+', ' ', x).replace(" "," ").replace("  "," ").strip())
print(test_set)

# Buid an MXN matrix where M is the number of samples, N is the number of unique words in the corpus (subject to parameters) and element [i,j] is the
# whether or not sample i contains word j
count_vectorizer = CountVectorizer()

# need to be everything!
full=list(job_titles.job_title)+list(test_set["Job Title"])
count_vectorizer.fit(full)
# print(count_vectorizer)
X = count_vectorizer.transform(job_titles.job_title)
print(X)
# Dump results into a pandas DataFrame since this is a small example for illustrative purposes
df = pd.DataFrame(X.toarray(), columns=count_vectorizer.get_feature_names())

print(df)
# Now consider a new title
# read the sample of the test 
X_new = count_vectorizer.transform(list(test_set["Job Title"]))
df_new = pd.DataFrame(X_new.toarray(), columns=count_vectorizer.get_feature_names())
print(df_new)

# # # # # # # # # 
# NB model fitting
# # # # # # # # # 
naive_bayes = BernoulliNB(alpha=1)  # make alpha 1
naive_bayes.fit(X=df, y=y)
t=naive_bayes.predict_proba(X=df_new)


# # # # # # # # # 
# getting predictions by NB model
# # # # # # # # # 
import operator
rows=[]
for i in range(len(test_set.index)):
    my_list=t[i]
    index, value = max(enumerate(my_list), key=operator.itemgetter(1))
    print(i, test_set["Job Title"][i],"||||||||",value, h[index])
    rows.append([i,test_set["Job Title"][i],value,h[index],index])

NB=pd.DataFrame(rows,columns=["sn","job_title","probablity","deparment","deparment_code"])
NB.to_csv("data/test/test_results.csv")

# # # # # # # # # 
# benchmarking with manual verification
# # # # # # # # # 

# load the manually classification:
manual =pd.read_csv("data/test/jobtitles_manual.csv")[:100]
manual["departments.1"]
# get the accuracy rate~

j=0
for i in manual.index:
    print((i,manual["Job Title"][i],"|||",NB["job_title"][i]))
    if manual["departments.1"][i]==NB["deparment"][i]:
        j+=1
        print(rows[i],manual["departments.1"][i],NB["deparment"][i])
print("accuracy rate: ",j)
# vs rule based