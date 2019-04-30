
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics


# In[2]:


## User data preparation for clustering
orders = pd.read_csv("Instacart_data/orders.csv")
countById = orders.groupby(['user_id']).count()
meanById = orders.groupby(['user_id']).mean()
userCountDF = countById[['order_id']].copy().rename(index=str, columns={"order_id": "order_count"})
userMeanDF = meanById[['order_dow', 'order_hour_of_day', 'days_since_prior_order']].copy().rename(index=str, columns={'order_dow' : 'avg_order_day', 'order_hour_of_day': 'avg_order_hour', 'days_since_prior_order' : 'avg_days_between_orders'}).round(0)
userDF = userCountDF.join(userMeanDF)


# In[3]:


## KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(userDF)
labels = kmeans.labels_
userDF = userDF.join(pd.DataFrame(labels, index=userDF.index, columns=['cluster']))


# In[5]:


## Create list of products, each product represented as a list of words (create list of lists)
products = pd.read_csv("Instacart_data/products.csv")
productNames = products[['product_name']].values
splitProducts = []
for product in productNames:
    productName = product[0].lower()
    productList = productName.split()
    splitProducts += [productList]


# In[6]:


products_aisle_dept = products[['product_id','aisle_id', 'department_id']]
products_aisle_dept.set_index('product_id', inplace=True)


# In[7]:


# Load Google's pre-trained Word2Vec model.
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)


# In[8]:


## Get embeddings for each product by averaging embeddings of component words, store in dataframe
def dict_retriever(product):
    if product.lower() in prodVectorDict:
        return prodVectorDict[product.lower()]
    else:
        return np.ones(300)

prodVectorDict = {}
for product in splitProducts:
    if product[0] in model.vocab:
        combinedVector = model[product[0]]
    else:
        combinedVector = combinedVector + np.ones(300)
    for word in product[1:]:
        if word in model.vocab:
            combinedVector = combinedVector + model[word]
        else:
            combinedVector = combinedVector + np.ones(300)
    averageVector = combinedVector/len(product)
    productString = ' '.join(product)
    prodVectorDict[productString] = averageVector
prodVectorDF = pd.DataFrame(products[['product_name', 'product_id']])
prodVectorDF['product_embedding'] = prodVectorDF['product_name'].apply(lambda x: dict_retriever(x))
prodVectorDF = prodVectorDF.set_index('product_id')


# In[9]:


prodVectorDF = prodVectorDF.join(products_aisle_dept)


# In[10]:


## Code to generate subcart embeddings for all our known subcarts


## Function taken and slightly adapted from https://gist.github.com/jlln/338b4b0b55bd6984f883.
def splitDataFrameList(df,target_column,separator):
    ''' df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column]
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df

orderDetailsDF = pd.read_csv("Instacart_data/order_products__prior.csv")
suborderDetailsDF = pd.DataFrame(orderDetailsDF.groupby('order_id')['product_id'].apply(list))
suborderDetailsDF['product_id'] = suborderDetailsDF['product_id'].map(lambda x: x[:len(x)-1])
suborderDetailsDF = suborderDetailsDF.reset_index()
suborderDetailsDF = splitDataFrameList(suborderDetailsDF,'product_id', ',')
suborderDetailsDF = suborderDetailsDF.join(prodVectorDF, on="product_id")
subcartEmbeddingDF = pd.DataFrame(suborderDetailsDF.groupby(['order_id'])['product_embedding'].apply(np.mean))
subcartEmbeddingDF = subcartEmbeddingDF.rename(columns={"product_embedding": "subcart_embedding"})


# In[11]:


## Read and format positive training examples
orders = pd.read_csv("Instacart_data/orders.csv")
countById = orders.groupby(['user_id']).count()
frequentBuyers = countById[countById['order_id'] >= 10]   ## Only users with more than 10 purchases considered
sampleOrders = pd.read_csv("Instacart_data/order_products__prior.csv")
userSample50 = frequentBuyers.sample(n=30000)             ## Sample users to reduce size of dataset
userSample = orders[orders.user_id.isin(userSample50.index.values)]
orderDF = pd.DataFrame(sampleOrders.groupby('order_id')['product_id'].apply(list))
orderDF = orderDF.rename(index=str, columns={"product_id": "order"})
orderDF.index = orderDF.index.map(int)
usersDF = pd.read_csv("Instacart_data/orders.csv")
usersDF = usersDF[['order_id', 'user_id']].set_index('order_id')
usersDF.index = usersDF.index.map(int)
userOrderDF = orderDF.join(usersDF, how="inner")
userOrderDF['keeporder'] = userOrderDF['order'].map(lambda x: len(x))
userOrderDF = userOrderDF[userOrderDF['keeporder'] > 10]
userOrderDF = userOrderDF.drop(['keeporder'], axis=1)
userOrderDF['suborder'] = userOrderDF['order'].map(lambda x: x[:len(x)-1])
userOrderDF['next_item'] = userOrderDF['order'].map(lambda x: x[len(x)-1])
userOrderDF['label'] = [True] * len(userOrderDF)
userOrderDF.reset_index(inplace=True)
userOrderDF.rename(index=str, columns={"index": "order_id"},inplace=True)


# In[12]:


## These 100 true samples are used later for pytrec, removing here was the easiest way
true_subset = userOrderDF.sample(n=100, random_state=42)
userOrderDF.drop(true_subset.index, inplace=True)


# In[13]:


## Negative sampling by randomly assigning false next_items at a ratio of 4:1 with true samples
products = pd.read_csv("Instacart_data/products.csv")
productlist = products['product_id'].tolist()
userOrderDF = userOrderDF.reset_index()
falseExamples = pd.DataFrame()
falseExamples = falseExamples.append([userOrderDF]*4)
falseExamples = falseExamples.rename(index=str, columns={"next_item": "correct_item"})
falseExamples['next_item'] = falseExamples['correct_item'].map(lambda x: np.random.choice(productlist, 1) )
falseExamples['label'] = [False] * len(falseExamples)
falseExamples.drop(['correct_item'], axis=1, inplace=True)
falseExamples['next_item'] = falseExamples['next_item'].map(lambda x: x[0])


# In[14]:


## Do before pickling
falseExamples.reset_index(inplace=True)
falseExamples.drop(['index','level_0'], axis=1,inplace=True)


# In[19]:


import pickle
falseDict = falseExamples.to_dict()
with open('false.pickle', 'wb') as handle:
    pickle.dump(falseDict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[13]:


import pickle
with open('false.pickle', 'rb') as handle:
    falseDict = pickle.load(handle)
    
falseExamples = pd.DataFrame.from_dict(falseDict)


# In[ ]:


## RERUN EARLY CELLS AND RESTART THIS, SHOULD BE GOOD FOR LATER


# In[14]:


## Create dataset by joining positive and negative samples
datasetDF = pd.concat([falseExamples,userOrderDF])


# In[127]:


datasetDF.drop(['index', 'level_0'], axis=1, inplace=True)
true_subset.drop(['index', 'level_0'], axis=1, inplace=True)


# In[17]:


## Append features relating to user
datasetDF['user_id'] = pd.to_numeric(datasetDF['user_id'])
userDF.index = pd.to_numeric(userDF.index)
userFeatures = datasetDF.join(userDF, on="user_id", how="inner")
userFeatures.rename(index=str, columns={"cluster": "user_cluster"},inplace=True)


# In[18]:


## Append features relating to next product

## Just to create column names for embedding values
column_headers= []
for i in range(1,301):
    header = "prod_embed_" + str(i)
    column_headers += [header]
    
    
prodVectorDF = prodVectorDF.drop(['product_name'], axis=1)
prodVectorDF[column_headers] = pd.DataFrame(prodVectorDF.product_embedding.values.tolist(), index= prodVectorDF.index)
prodVectorDF = prodVectorDF.drop(['product_embedding'], axis=1)
userProdFeatures = userFeatures.join(prodVectorDF, on="next_item", how="inner")


# In[19]:


## Append features relating to current cart (subcart)

cart_column_headers= []
for i in range(1,301):
    header = "subcart_embed_" + str(i)
    cart_column_headers += [header]
    
userProdFeatures['order_id'] = pd.to_numeric(userProdFeatures['order_id'])
subcartEmbeddingDF[cart_column_headers] = pd.DataFrame(subcartEmbeddingDF.subcart_embedding.values.tolist(), index= subcartEmbeddingDF.index)
subcartEmbeddingDF = subcartEmbeddingDF.drop(['subcart_embedding'], axis=1)
subEmbSamp = subcartEmbeddingDF.sample(320000, random_state=42)   ## Random sampling just to reduce size of dataset
finalDataset = userProdFeatures.join(subEmbSamp, on="order_id", how="inner")


# In[21]:


## Tidy up final dataset and extract labels
#finalDataset['label'] = finalDataset['label'].map(lambda x: 1 if x else 0)
labels = finalDataset['label']
finalDataset.drop(['next_item', 'order', 'suborder', 'order_id', 'user_id','label'], axis=1, inplace=True)
finalDataset = finalDataset.reset_index()
finalDataset.drop(['index'], axis=1, inplace=True)


# In[35]:


## split generated data into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(finalDataset, labels, test_size=0.20, random_state=42)


# In[58]:


## Fit MLP neural network
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(max_iter=30).fit(X_train, y_train)


# In[36]:


## Fit Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)


# In[37]:


## Fit LinearSVM classifier
from sklearn import linear_model
clf = linear_model.SGDClassifier(loss="hinge").fit(X_train,y_train)


# In[38]:


## Generate dataset for pytrec evaluation, its cumbersome doing this twice but ensures
## test samples are totally separate than training samples
false_subset = pd.DataFrame()
false_subset = false_subset.append([true_subset]*100)
false_subset['next_item'] = false_subset['next_item'].map(lambda x: np.random.choice(productlist, 1))
false_subset['label'] = [False] * len(false_subset)
false_subset['next_item'] = false_subset['next_item'].map(lambda x: x[0])
pytrecDataset = pd.concat([false_subset,true_subset])
pytrecDataset.reset_index(inplace=True)
pytrecDataset.drop(['index'], axis=1, inplace=True)


# In[1]:


import pickle


# In[3]:


with open('pytrec_eval/logreg_run.pickle', 'rb') as handle:
    run = pickle.load(handle, encoding='latin1')
with open('pytrec_eval/logreg_qrel.pickle', 'rb') as handle:
    qrel = pickle.load(handle, encoding='latin1')


# In[80]:


orderids = np.unique(pytrecDataset.order_id.values)
qrel = {}
run = {}
for i in range(0, len(orderids)):
    thisTest = pytrecDataset.loc[pytrecDataset["order_id"] == orderids[i]]
    thisTest['user_id'] = pd.to_numeric(thisTest['user_id'])
    userDF.index = pd.to_numeric(userDF.index)
    withUserFeatures = thisTest.join(userDF, on="user_id", how="inner")
    withProdFeatures = withUserFeatures.join(prodVectorDF, on="next_item", how="inner")
    thisFinal = withProdFeatures.join(subcartEmbeddingDF, on="order_id", how="inner")
    thisFinal['label'] = thisFinal['label'].map(lambda x: 1 if x else 0)
    thisFinalLabels = thisFinal['label']
    thisFinal = thisFinal.drop(['next_item', 'order', 'suborder', 'order_id', 'user_id','label'], axis=1)
    ##preds = clf.decision_function(thisFinal)
    preds = logreg.predict_proba(thisFinal)
    #preds = nn.predict_proba(thisFinal)
    confidence_dict = {}
    truth_dict = {}
    for x in range(0,len(preds)):
        confidence_dict[str(x)] = abs(preds[x][1])
        truth_dict[str(x)] = int(thisFinalLabels.values[x])
    qrel[str(i)] = truth_dict
    run[str(i)] = confidence_dict
    


# In[62]:


import pytrec_eval
import json

def Average(lst): 
    return sum(lst) / len(lst)
evaluator = pytrec_eval.RelevanceEvaluator(qrel,{'map', 'ndcg'})
results = evaluator.evaluate(run)
NDCGlist = []
MAPlist = []
for key in results:
    result = results[key]
    NDCGlist += [result['ndcg']]
    MAPlist += [result['map']]
print ("==========================Results=============================")
print ("Average NDCG: " + str(Average(NDCGlist)))
print ("MAP: " + str(Average(MAPlist)))
print ("==============================================================")

