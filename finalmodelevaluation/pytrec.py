
# coding: utf-8

# In[1]:


import pickle
import pytrec_eval
import json


# In[2]:


def get_qrels(modelName):
    with open('%s_qrel.pickle' %modelName, 'rb') as handle:
        qrel = pickle.load(handle, encoding='latin1')
    return qrel


# In[3]:


def get_run(modelName):
    with open('%s_run.pickle' %modelName, 'rb') as handle:
        run = pickle.load(handle, encoding='latin1')
    return run


# In[24]:


implicit_qrel = get_qrels('implicit')
implicit_run = get_run('implicit')
scipy_qrel = get_qrels('scipy_recs')
scipy_run = get_run('scipy_recs')
spotseq_qrel = get_qrels('scipy')
spotseq_run = get_run('scipy')
support_qrel = get_qrels('support')
support_run = get_run('support')
listwise_map_qrel = get_qrels('listwise_map')
listwise_map_run = get_run('listwise_map')
listwise_ndcg_qrel = get_qrels('listwise_ndcg')
listwise_ndcg_run = get_run('listwise_ndcg')
pairwise_qrel = get_qrels('pairwise')
pairwise_run = get_run('pairwise')


# In[47]:


implicit_evaluator = pytrec_eval.RelevanceEvaluator(implicit_qrel,{'map', 'ndcg'})
implicit_results = implicit_evaluator.evaluate(implicit_run)


# In[28]:


scipy_evaluator = pytrec_eval.RelevanceEvaluator(scipy_qrel,{'map', 'ndcg'})
scipy_results = scipy_evaluator.evaluate(scipy_run)


# In[73]:


spotseq_evaluator = pytrec_eval.RelevanceEvaluator(spotseq_qrel,{'map', 'ndcg'})
spotseq_results = spotseq_evaluator.evaluate(spotseq_run)


# In[77]:


support_evaluator = pytrec_eval.RelevanceEvaluator(support_qrel,{'map', 'ndcg'})
support_results = support_evaluator.evaluate(support_run)


# In[10]:


listwise_map_evaluator = pytrec_eval.RelevanceEvaluator(listwise_map_qrel,{'map', 'ndcg'})
listwise_map_results = listwise_map_evaluator.evaluate(listwise_map_run)


# In[ ]:


listwise_ndcg_evaluator = pytrec_eval.RelevanceEvaluator(listwise_ndcg_qrel,{'map', 'ndcg'})
listwise_ndcg_results = listwise_map_evaluator.evaluate(listwise_ndcg_run)


# In[ ]:


pairwise_evaluator = pytrec_eval.RelevanceEvaluator(pairwise_qrel,{'map', 'ndcg'})
pairwise_results = pairwise_evaluator.evaluate(pairwise_run)


# In[6]:


def Average(lst): 
    return sum(lst) / len(lst)


# In[7]:


def print_scores(results, modelName):
    NDCGlist = []
    MAPlist = []
    for key in results:
        result = results[key]
        NDCGlist += [result['ndcg']]
        MAPlist += [result['map']]
    print ("==========================" + modelName + " Results=============================")
    print ("Average NDCG: " + str(Average(NDCGlist)))
    print ("MAP: " + str(Average(MAPlist)))
    print ("==============================================================")


# In[48]:


print_scores(implicit_results, "Implicit")

