import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer



df1 = pd.read_csv("shared_articles.csv") 
df2 = pd.read_csv("users_interactions.csv") 


df1 = df1[df1['eventType'] == "CONTENT SHARED"]

def total_events(df1_row) :
    total_likes = df2[(df2["contentId"] == df1_row["contentId"]) & (df2["eventType"] == "LIKE")].shape[0]
    total_views = df2[(df2[(df2["contentId"] == df1_row["contentId"]) & (df2["eventType"] == "VIEW")])].shape[0]
    total_bookmarks = df2[(df2[(df2["contentId"] == df1_row["contentId"]) & (df2["eventType"] == "BOOKMARK")])].shape[0]
    total_follows = df2[(df2[(df2["contentId"] == df1_row["contentId"]) & (df2["eventType"] == "FOLLOW")])].shape[0]
    total_comment_created = df2[(df2[(df2["contentId"] == df1_row["contentId"]) & (df2["eventType"] == "COMMENT CREATED")])].shape[0]

    return total_likes + total_views + total_bookmarks + total_comment_created + total_follows



df1["total_events"] = df1.apply(total_events, axis = 1)

df1 = df1.sort_values(["total_events"], ascending = [False])
print(df1.head())


count = CountVectorizer(stop_words = 'english')
count_matrix = count.fit_transform(df1['title'])

cosine = cosine_similarity(count_matrix, count_matrix)

df1 = df1.reset_index()
indices = pd.Series(df1.index, index = df1['contentId'])

def get_recommendations(contentId, cos):
  id = indices[contentId]
  simi_scr = list(enumerate(cos[id]))
  simi_scr = sorted(simi_scr, key=lambda x: x[1], reverse = True)
  simi_scr = simi_scr[1:11]
  article = [i[0] for i in simi_scr]

  return df1[["url", "title", "text", "lang", "total_events"]].iloc[article]

print(get_recommendations(-4029704725707465084, cosine))
