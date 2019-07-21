#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import pandas as pd
import json
import pprint
from collections import Counter

#load sample deparmtent data
f=open('data/departments.json') 
x = ast.literal_eval(f.read().lower())
k=[list(item.keys())[0] for item in x]

# print(k)
dictionary={}
i=0
for item in k:
	dictionary[item]=i
	i+=1
pprint.pprint(dictionary)

departments={}
for item in x:
	departments[list(item.keys())[0]]=item[list(item.keys())[0]]

pprint.pprint(x)


#load raw job title
title =pd.read_csv("data/jobtitles_traing.csv",encoding='ISO-8859–1')

to_remove=[]
title["title_cleaned"]=title["Job Title"].str.lower()

title_list=list(title["title_cleaned"])
title_list=[str(item) for item in title_list]
t=" ".join(title_list)
Counter(t.split(" ")).most_common()
# title =pd.read_csv("data/jobtitles.csv",encoding='ISO-8859–1')
# print(len(title.index))

# title_test=title.sample(300)
# title_test.to_csv("data/jobtitles_test.csv")

# title=title[~title.index.isin(list(title_test.index))]
# title.to_csv("data/jobtitles_traing.csv")
# print(len(title.index))

# sleep(1000)


raw_job_titles=[]



csv_rows=[]

for department in k:
	print("-----")
	print(department)

	t=title[title["Job Title"].apply(lambda x: any([item in str(x) for item in departments[department]]))]
	for i in t.index:
		csv_rows.append([t["Job Title"][i],department])

pd.DataFrame(csv_rows,columns=["job_title","job_category"]).to_csv("data/training/job_titles_with_labels.csv")