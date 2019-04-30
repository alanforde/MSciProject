
# coding: utf-8

# ## Contents:
# * [Set Up](#setup)
# * [Support](#support)
# * [Implicit](#implicit)
# * [SciPy](#scipy)
# * [Spotlight Sequence Recommender](#spot_seq)
# * [Spotlight ALS](#spot)
# * [Features](#features)
# * [Evaluation](#eval)

# In[1]:


import pandas as pd
import numpy as np
import scipy


# ======================= SET UP ======================= <a class="anchor" id="setup"></a>

# In[2]:


ORDER_SAMPLE_SIZE = 1000 #Number of orders we want to use a sample
MIN_PURCHASES = 30 #How many purchases needed per order to be included in sample


# In[3]:


purchases = pd.read_csv("Instacart_data/order_products__prior.csv")
purchases.drop(['add_to_cart_order', 'reordered'], axis=1, inplace=True)


# In[4]:


products = pd.read_csv("Instacart_data/products.csv")
products.drop(['aisle_id', 'department_id'], axis=1, inplace=True)


# In[5]:


orders = pd.read_csv("Instacart_data/orders.csv")
orders.drop(['eval_set', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'], axis=1, inplace=True)
orders.set_index('order_id', inplace=True)


# In[6]:


purchases = purchases.join(orders, on='order_id')
purchases['purchased'] = np.ones(len(purchases))


# In[7]:


purchases.drop_duplicates(inplace=True)


# In[8]:


listed_df = pd.DataFrame(purchases.groupby('order_id')['product_id'].apply(list))


# In[9]:


listed_df['test_prod'] = listed_df.product_id.apply(lambda x: x[-1])
listed_df['product_id'] = listed_df.product_id.apply(lambda x: x[:-1])


# In[10]:


lastInBasket = listed_df.drop('product_id', axis=1).rename(columns = {'test_prod': 'product_id'})


# In[11]:


purchases2 = (listed_df.product_id.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('product_id'))


# In[12]:


purchases = purchases2.copy()


# In[13]:


purchases.reset_index(inplace=True)
purchaseCount = purchases.groupby('order_id').count()
largeOrders = purchaseCount[purchaseCount['product_id'] >= MIN_PURCHASES]
largeOrders = largeOrders.sample(ORDER_SAMPLE_SIZE, random_state=42)
purchases = purchases.loc[purchases.order_id.isin(largeOrders.index.values)]


# In[14]:


purchases['purchased'] = np.ones(len(purchases))


# In[71]:


equiv_id_orders=  {}
order = 0
for order_id in np.unique(purchases.order_id.values):
    equiv_id_orders[order_id] = order
    order += 1


# In[72]:


equiv_id_prods =  {}
prod = 1
for product_id in np.unique(purchases.product_id.values):
    equiv_id_prods[product_id] = prod
    prod += 1


# In[133]:


implicit_equiv_id_prods =  {}
prod = 0
for product_id in np.unique(purchasesWithFinal.product_id.values):
    implicit_equiv_id_prods[product_id] = prod
    prod += 1


# In[134]:


implicit_equiv_prod_DF = pd.DataFrame.from_dict(implicit_equiv_id_prods, orient='index')
implicit_equiv_prod_DF.rename(columns = {0: 'equiv_product_id'}, inplace=True)


# In[70]:


## If we want to ensure the next product purchased by every order is infact in the model
lastProducts = lastInBasket.reset_index()
lastProducts = lastProducts[lastProducts.order_id.isin(purchases.order_id.values)]
lastProducts['order_id'] = lastProducts.order_id.apply(lambda x: 1)
lastProducts['purchased'] = np.ones(len(lastProducts))
lastProducts.drop_duplicates(inplace=True)
purchasesWithFinal = pd.concat([lastProducts, purchases])
purchasesWithFinal.reset_index(inplace=True)
purchasesWithFinal.drop(['index'], axis=1, inplace=True)


# ======================= SUPPORT ======================= <a class="anchor" id="support"></a>

# In[19]:


productCount = purchases.groupby('product_id').count().sort_values('order_id', ascending=0).reset_index()


# In[20]:


productCount.reset_index(inplace=True)
productCount['pop_rank'] = productCount['index'].apply(lambda x: x+1)


# In[21]:


supportDF = productCount['product_id']


# In[22]:


def getSupportRecs():
    recs_dict = {}
    for order in np.unique(purchases.order_id.values):
        recs_dict[order] = supportDF.values
    return recs_dict


# In[23]:


support_recs = getSupportRecs()


# ======================= IMPLICIT ======================= <a class="anchor" id="implicit"></a>

# In[24]:


import implicit


# In[85]:


implicit_userPurchases = purchasesWithFinal.pivot(index = 'product_id', columns ='order_id', values = 'purchased').fillna(0)


# In[86]:


implicit_purchaseMatrix = implicit_userPurchases.as_matrix()


# In[87]:


implicit_purchaseMatrix = scipy.sparse.csr_matrix(implicit_userPurchases.values)


# In[88]:


implicit_als_model = implicit.als.AlternatingLeastSquares(factors=100)


# In[89]:


implicit_als_model.fit(implicit_purchaseMatrix)


# In[90]:


user_items = implicit_purchaseMatrix.T.tocsr()


# In[127]:


top100real = support_recs[546][:100]    ##arbitrary ID, support_recs are the same for everyone
top100equiv = []
for product in top100real:
    top100equiv += [implicit_equiv_prod_DF.loc[product].values[0]]


# In[131]:


def get_implicit_recs():
    recs_dict = {}
    for order in np.unique(purchases.order_id.values):
        trueNext = lastInBasket.loc[order].values[0]
        equivTruthId = implicit_equiv_prod_DF.loc[trueNext].values[0]
        items2rank = top100equiv[:]
        items2rank += [equivTruthId]
        recommendations = implicit_als_model.rank_items(equiv_id_orders[order], user_items, items2rank)
        recs_list = []
        for rec in recommendations:
            recs_list += [implicit_equiv_prod_DF.loc[implicit_equiv_prod_DF.equiv_product_id == rec[0]].index.values[0]]
        recs_dict[order] = recs_list
    return recs_dict


# In[129]:


recs_dict = {}
for order in [546]:#np.unique(purchases.order_id.values):
    trueNext = lastInBasket.loc[order].values[0]
    equivTruthId = implicit_equiv_prod_DF.loc[trueNext].values[0]
    items2rank = top100equiv[:]
    items2rank += [equivTruthId]
    recommendations = implicit_als_model.rank_items(equiv_id_orders[order], user_items, items2rank)
    recs_list = []
    for rec in recommendations:
        recs_list += [implicit_equiv_prod_DF.loc[implicit_equiv_prod_DF.equiv_product_id == rec[0]].index.values[0]]
    recs_dict[order] = recs_list


# In[135]:


implicit_recs = get_implicit_recs()


# ======================= SCIPY ======================= <a class="anchor" id="scipy"></a>

# In[18]:


scipy_userPurchases = purchases.pivot(index = 'order_id', columns ='product_id', values = 'purchased').fillna(0)


# In[19]:


scipy_purchaseMatrix = scipy_userPurchases.as_matrix()


# In[20]:


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(scipy_purchaseMatrix, k = 50)


# In[21]:


sigma = np.diag(sigma)


# In[22]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
preds_df = pd.DataFrame(all_user_predicted_ratings, index = scipy_userPurchases.index.values, columns = scipy_userPurchases.columns)


# In[23]:


def get_scipy_recs():
    recs_dict = {}
    for order in np.unique(purchases.order_id.values):
        sorted_order_predictions = preds_df.loc[order].sort_values(ascending=False)
        order_data = purchases[purchases.order_id == (order)]
        ##filtered_preds = (order_data.merge(products, how = 'left', left_on = 'product_id', right_on = 'product_id').sort_values(['purchased'], ascending=False))
        recommendations = (products[~products['product_id'].isin(order_data['product_id'])].
         merge(pd.DataFrame(sorted_order_predictions).reset_index(), how = 'left',
           left_on = 'product_id',
           right_on = 'product_id').rename(columns = {order: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                   iloc[:10044, :-1]
                  )
        recs_dict[order] = recommendations.product_id.values
    return recs_dict
        


# In[24]:


scipy_recs = get_scipy_recs()


# ======================= SPOTLIGHT SEQUENCE RECOMMENDER ======================= <a class="anchor" id="spot_seq"></a>

# In[40]:


from spotlight.cross_validation import random_train_test_split
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight import interactions


# In[41]:


withTimes = pd.read_csv("Instacart_data/order_products__prior.csv")
withTimes.drop(['reordered'], axis=1, inplace=True)


# In[42]:


purchasesWithTimes = pd.merge(purchases, withTimes,  how='left', left_on=['order_id','product_id'], right_on = ['order_id','product_id'])


# In[43]:


purchasesWithTimes['equiv_prod_id'] = purchasesWithTimes.product_id.apply(lambda x: equiv_id_prods[x])
purchasesWithTimes['equiv_order_id'] = purchasesWithTimes.order_id.apply(lambda x: equiv_id_orders[x])


# In[44]:


orders = purchasesWithTimes.equiv_order_id.values.astype(np.int32)
prods = purchasesWithTimes.equiv_prod_id.values.astype(np.int32)
timestamps = purchasesWithTimes.add_to_cart_order.values.astype(np.int32)


# In[45]:


dataset = interactions.Interactions(orders, prods, timestamps=timestamps)
dataset = dataset.to_sequence()


# In[46]:


spotlight_seq_model = ImplicitSequenceModel(n_iter=10,
                              representation='cnn',
                              loss='bpr')


# In[47]:


spotlight_seq_model.fit(dataset)


# In[52]:


def get_spotlight_seq_recs():
    recs_dict = {}
    for order in np.unique(purchases.order_id.values):
        recs_list = []
        productScores = spotlight_seq_model.predict(purchasesWithTimes.loc[purchasesWithTimes.order_id == order, 'equiv_prod_id'].values)
        recsDF = pd.DataFrame({'product':np.unique(purchasesWithTimes.equiv_prod_id.values), 'score':productScores[1:]})
        recsDF.sort_values('score', ascending=0, inplace=True)
        recommendations = recsDF['product'].values[:1000]
        for rec in recommendations:
            recs_list += [implicit_equiv_prod_DF.loc[implicit_equiv_prod_DF.equiv_product_id == rec-1].index.values[0]]
        recs_dict[order] = recs_list
    return recs_dict


# In[53]:


spotlight_seq_recs = get_spotlight_seq_recs()


# ======================= SPOTLIGHT ======================= <a class="anchor" id="spot"></a>

# In[46]:


from spotlight.cross_validation import random_train_test_split
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight import interactions


# In[69]:


spotlight_data = interactions.Interactions(orders, prods)


# In[70]:


spotlight_ifm_model = ImplicitFactorizationModel(n_iter=10,
                                   loss='bpr')


# In[71]:


spotlight_ifm_model.fit(spotlight_data)


# In[74]:


def get_spotlight_recs():
    recs_dict = {}
    for order in np.unique(purchases.order_id.values):
        recommendations = spotlight_ifm_model.predict(purchasesWithTimes.loc[purchasesWithTimes.equiv_order_id == equiv_id_orders[order], 'equiv_prod_id'].values)
        recs_dict[order] = recommendations
    return recs_dict


# In[75]:


spotlight_recs = get_spotlight_recs()


# ======================= FEATURES =======================  <a class="anchor" id="features"></a>

# In[25]:


from sklearn.cluster import KMeans
import gensim
import string


# In[26]:


model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)


# In[27]:


## Build one-hot encoding dataframe
productsNames = pd.read_csv("Instacart_data/products.csv")
productNames = products[['product_name']].values
splitProducts = []
for product in productNames:
    productName = product[0].lower()
    productList = productName.split()
    splitProducts += [productList]
productNamesDF = products[['product_name']]
vocab = np.concatenate(splitProducts)
vocab = np.unique(vocab)
count = 0 
for prod in vocab:
    productNamesDF[prod] = productNamesDF['product_name'].apply(lambda x: 1 if prod in x else 0)
    count += 1
    if (count%1000 == 0): print (count)


# In[27]:


## Build product embedding dataframe by averaging embeddings of component words
products_df = pd.read_csv("Instacart_data/products.csv")
productNames = products_df[['product_name']].values
products_df['split_product_name'] = products_df.product_name.apply(lambda x: x.split())
stacked_products_df  = (products_df.split_product_name.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('product_words'))
stacked_products_df['product_words'] = stacked_products_df.product_words.apply(lambda x: x.translate(None, string.punctuation).lower())
products_df.reset_index(inplace=True)
products_df = products_df.join(stacked_products_df, on='index')
column_headers= []
for i in range(1,301):
    header = "prod_embed_" + str(i)
    column_headers += [header]
products_df["word_embedding"] = pd.DataFrame(products_df.product_words.apply(lambda x: model[x] if x in model.vocab else np.ones(300)))
products_df[column_headers] = pd.DataFrame(products_df.word_embedding.values.tolist(), index= products_df.index)
product_embeddings_df = products_df.groupby('product_id').mean().drop(['index'], axis=1)
product_embeddings_df.drop(['aisle_id', 'department_id'], axis=1, inplace=True)


# In[28]:


prod_column_headers= []
for i in range(1,301):
    header = "prod_embed_" + str(i)
    prod_column_headers += [header]


# In[29]:


order_column_headers= []
for i in range(1,301):
    header = "basket_embed_" + str(i)
    order_column_headers += [header]


# In[30]:


productHeaderToOrderHeader = {}
for header in range(0, len(prod_column_headers)):
    productHeaderToOrderHeader[prod_column_headers[header]] = order_column_headers[header]


# In[31]:


NUM_RECS=1000


# In[32]:


purchase_data = listed_df.copy()
purchase_data.reset_index(inplace=True)
purchase_data.rename(columns = {'product_id': 'basket', 'test_prod':'nextInBasket'}, inplace=True)


# In[33]:


## Get additional information about user/basket so that features can be computed
def get_order_info_df(recs_dict):
    orders = []
    for order in recs_dict:
        orders += [purchase_data.loc[purchase_data.order_id == order]]
    order_info_df = pd.concat(orders, ignore_index=True)
    order_info_df['recommendation'] = order_info_df.order_id.apply(lambda x: recs_dict[x][:NUM_RECS])
    order_info_df.set_index('order_id', inplace=True)
    order_info_df2 = (order_info_df.recommendation.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('recommendation'))
    order_info_df.drop(['recommendation'], axis=1, inplace = True)
    order_info_df = order_info_df.join(order_info_df2)
    order_info_df['label'] = order_info_df.nextInBasket == order_info_df.recommendation
    return order_info_df


# In[34]:


#implicit_order_info = get_order_info_df(implicit_recs)
scipy_order_info = get_order_info_df(scipy_recs)
#spotseq_order_info = get_order_info_df(spotlight_seq_recs)
#support_order_info = get_order_info_df(support_recs)


# In[35]:


def appendUserFeatures(order_info_df):
    order_info = order_info_df.copy()
    
    ## Basic Features; order count, average order day, hour and time between orders
    orders = pd.read_csv("Instacart_data/orders.csv")
    countById = orders.groupby(['user_id']).count()
    meanById = orders.groupby(['user_id']).mean()
    userCountDF = countById[['order_id']].copy().rename(index=str, columns={"order_id": "order_count"})
    userMeanDF = meanById[['order_dow', 'order_hour_of_day', 'days_since_prior_order']].copy().rename(index=str, columns={'order_dow' : 'avg_order_day', 'order_hour_of_day': 'avg_order_hour', 'days_since_prior_order' : 'avg_days_between_orders'}).round(0)
    userDF = userCountDF.join(userMeanDF)

    ## KMeans Clustering
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(userDF)
    labels = kmeans.labels_
    userDF = userDF.join(pd.DataFrame(labels, index=userDF.index, columns=['cluster']))
    userDF.set_index(userDF.index.astype(np.int32), inplace=True)

    ## Average Aisle and Department User shops in
    products_aisle_dept = pd.read_csv("Instacart_data/products.csv")
    products_aisle_dept = products_aisle_dept[['product_id','aisle_id', 'department_id']].copy()
    products_aisle_dept.set_index('product_id', inplace=True)
    user_aisle_dept = pd.read_csv("Instacart_data/order_products__prior.csv")
    user_aisle_dept = user_aisle_dept.join(orders.set_index('order_id'), on='order_id')
    user_aisle_dept = user_aisle_dept[['order_id', 'product_id', 'user_id']].copy()
    user_aisle_dept = user_aisle_dept.join(products_aisle_dept, on='product_id')
    user_aisle_dept =  user_aisle_dept.groupby(['user_id']).mean()
    user_aisle_dept.drop(['order_id', 'product_id'], axis=1, inplace=True)
    user_aisle_dept.rename(columns={"aisle_id": "avg_user_aisle", "department_id":"avg_user_dept"}, inplace=True)

    ## Get user_id from order_id to join features
    order_info.reset_index(inplace=True)
    order_info = order_info.join(orders[['order_id', 'user_id']].set_index('order_id'), on='order_id')

    ## Join features
    order_info = order_info.join(userDF, on='user_id')
    order_info = order_info.join(user_aisle_dept, on='user_id')
    order_info.drop(['user_id'], axis=1, inplace=True)
    
    return order_info


# In[36]:


def appendRecommendedProductFeatures(order_info_df):
    order_info = order_info_df.copy()
    purchase_data = pd.read_csv("Instacart_data/order_products__prior.csv")
    order_data = pd.read_csv("Instacart_data/orders.csv")
    product_data = pd.read_csv("Instacart_data/products.csv")
    
    ## Aisle and Department of Product
    product_data.set_index('product_id', inplace=True)
    product_data.drop(['product_name'], axis=1, inplace=True)
    product_data.rename(columns={'aisle_id': 'R_aisle_id', 'department_id': 'R_department_id'})
    order_info.join(product_data, on='recommendation')
    
    ## Average hour, day, time added to cart and interval at which product is ordered
    productsPerOrder = purchase_data.join(order_data.set_index('order_id'), on='order_id')
    product_mean_stats = productsPerOrder.groupby('product_id').mean()
    product_mean_stats.drop(['order_id', 'reordered', 'user_id', 'order_number'], axis=1, inplace=True)
    product_mean_stats.rename(columns = {'add_to_cart_order': 'R_avg_order_in_cart', 'order_dow': 'R_avg_day_of_week', 'order_hour_of_day': 'R_avg_hour_of_day', 'days_since_prior_order': 'R_avg_interval'}, inplace=True)
    order_info = order_info.join(product_mean_stats, on='recommendation')
    
    ## Append word embeddings for each product
    order_info = order_info.join(product_embeddings_df, on='recommendation')
    

    ## Has user ever purchased this product before?
    productsPerUser = purchase_data.join(order_data.set_index('order_id'), on='order_id')
    productsPerUser = pd.DataFrame(productsPerUser.groupby('user_id')['product_id'].apply(list))
    userProductsPerOrder = productsPerUser.join(order_data.set_index('user_id'))
    userProductsPerOrder.drop(['eval_set', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'], axis=1, inplace=True)
    userProductsPerOrder.rename(columns = {'product_id': 'previous_purchases'}, inplace=True)
    userProductsPerOrder.set_index('order_id', inplace=True)
    order_info = order_info.join(userProductsPerOrder)
    order_info['rec_purchased_before'] = order_info.apply(lambda x: x.recommendation in x.previous_purchases, axis=1)
    order_info.drop(['previous_purchases'], inplace=True)
    ## one-hot encoding for product

    return order_info


# In[37]:


def appendBasketFeatures(order_info_df):
    order_info = order_info_df.copy()
    orders_data = pd.read_csv("Instacart_data/orders.csv")
    
    ## Basket Size
    order_info['B_size'] = order_info.basket.apply(lambda x: len(x))
    
    ## Basket day of week, time of day, and time since last order
    orders_data.set_index('order_id', inplace=True)
    order_info = order_info.join(orders_data)
    order_info.drop(['user_id', 'eval_set', 'order_number'], axis=1, inplace=True)
    order_info.rename(columns = {'order_dow': 'B_day_of_week', 'order_hour_of_day': 'B_hour_of_day', 'days_since_prior_order': 'B_time_since_last_order'}, inplace=True)
    
    ## Basket Embedding -- average of products in basket
    order_basket_df = order_info[['order_id','basket']]
    order_basket_df.set_index('order_id', inplace=True)
    order_basket_df = (order_basket_df.basket.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('basketProduct'))
    order_basket_df.reset_index(inplace=True)
    order_basket_df.drop_duplicates(inplace=True)
    embedded_order_basket_df = order_basket_df.join(product_embeddings_df, on='basketProduct')
    embedded_order_basket_df.reset_index(inplace=True)
    embedded_order_basket_df = embedded_order_basket_df.groupby('order_id').mean()
    embedded_order_basket_df.rename(columns = productHeaderToOrderHeader, inplace=True)
    order_info = order_info.join(embedded_order_basket_df, on='order_id')
    
    
    return order_info


# In[38]:


def appendFeatures(order_info_df):
    order_info = order_info_df.copy()
    order_info = appendUserFeatures(order_info)
    order_info = appendRecommendedProductFeatures(order_info)
    order_info = appendBasketFeatures(order_info)
    return order_info
    


# In[39]:


withFeatures = appendFeatures(scipy_order_info)


# ======================= XGBOOST =======================  <a class="anchor" id="xgboost"></a>

# In[41]:


import xgboost as xgb
import math


# In[42]:


xgb_labels = withFeatures[['label', 'order_id']]
xgb_data = withFeatures.drop(['label', 'basket', 'nextInBasket', 'recommendation'], axis=1)


# In[43]:


## Need to define our own to ensure each train/test case is a full set of recommendations for a user
def get_test_orders(order_ids, testsize=0.2):
    return np.random.choice(order_ids, size = int(math.floor(len(order_ids)*testsize)))


# In[44]:


order_ids = np.unique(xgb_labels.order_id.values)


# In[45]:


test_order_ids = get_test_orders(order_ids, 0.2)


# In[47]:


X_test = xgb_data[xgb_data.order_id.isin(test_order_ids)]
y_test = xgb_labels[xgb_labels.order_id.isin(test_order_ids)]
X_train = xgb_data[~xgb_data.order_id.isin(test_order_ids)]
y_train = xgb_labels[~xgb_labels.order_id.isin(test_order_ids)]


# In[49]:


X_train.drop(['order_id'], axis=1, inplace=True)
X_test.drop(['order_id'], axis=1, inplace=True)
y_train.drop(['order_id'], axis=1, inplace=True)
y_test.drop(['order_id'], axis=1, inplace=True)


# In[50]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, y_test)


# In[52]:


param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'rank:map' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)


# In[53]:


preds = bst.predict(dtest)


# In[54]:


predsDF = y_test.copy()
predsDF['pred_rating'] = preds
predsDF = predsDF.join(tester[['order_id', 'recommendation', 'nextInBasket']])
predsDF.sort_values(by=['order_id', 'pred_rating'], ascending=False, inplace=True)
ranking_positions = range(1,NUM_RECS + 1)
predsDF['ranking'] = ranking_positions * len(np.unique(predsDF.order_id.values))


# In[55]:


listed_preds_df = pd.DataFrame(predsDF.groupby('order_id')['recommendation'].apply(list))


# In[56]:


l2r_recs_dict = {}
for orderid in listed_preds_df.index.values:
    l2r_recs_dict[orderid] = listed_preds_df.loc[orderid].recommendation


# ======================= EVALUATION ======================= <a class="anchor" id="eval"></a>

# In[139]:


import pickle


# In[101]:


'''
Params:
predictions: predictions for ALL orders in the form {order_id:[product_ids ranked]}
K: Number of predictions to computer recall score at
'''
def calculate_recall_K(predictions, K):
    success_counter = 0.0
    for order in predictions:
        truth = lastInBasket.loc[order].values[0]
        preds = predictions[order][:K]
        for prod in preds:
            if prod == truth:
                success_counter += 1
                
    return success_counter/len(predictions)


# In[140]:


'''
Params:
predictions: predictions for ALL orders in the form {order_id:[product_ids ranked]}
K: Number of predictions to computer mAP score at
'''
def calculate_map_K(predictions, K):
    mAP= 0.0
    for order in predictions:
        counter = 0.0
        truth = lastInBasket.loc[order].values[0]
        preds = predictions[order][:K]
        for prod in range(len(preds)):
            if preds[prod] == truth:
                counter += 1
                mAP += (counter/(prod+1))
    return mAP/len(predictions)


# In[59]:


'''
Need to use Python3 for pytrec_eval, so we pickle recommendations
'''
def pickle_recs(predictions, modelName, numrecs):
    qrel = {}
    run = {}
    for order in predictions:
        pred_scores = {}
        true_scores = {}
        preds = predictions[order]
        for i in range(numrecs):
            pred_scores[str(preds[i])] = 1.0/(i+1)
            true_scores[str(preds[i])] = 1 if preds[i] == lastInBasket.loc[order].values[0] else 0
        run[str(order)] = pred_scores
        qrel[str(order)] = true_scores
    with open('%s_run.pickle' % modelName, 'wb') as handle:
        pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s_qrel.pickle' % modelName, 'wb') as handle:
        pickle.dump(qrel, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    return qrel, run


# In[144]:


calculate_map_K(implicit_recs, 100)


# In[108]:


calculate_recall_K(scipy_recs, 20000)

