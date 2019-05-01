# MSciProject
Code used in the implementation of several recommender systems in part fulfilment of an MSci Individual Project at University of Glasgow.


FinalModelEvaluation contains model_eval.*, where all the important things are. This file contains the formatting of the dataset and subsequent implementation of the four recommender systems, XGBoost model training and ranking, and some evaluation measures. The file pytrec.* is separate because it necessitated the use of Python3, and exists to return NDCG and MAP scores (although MAP is also implemented in model_eval.py).

InitalPOCsystem is just a classifier-based recommender system developed in first semester for the project proposal, to demonstrate the dataset was usable among other things.


Aside: Obviously these were written in Python notebooks, as such, each repository contains a .ipynb version of each file such that the documents can be reopened in a notebook and retain their cell structure. The .py files exist only so the code can be browsed on GH -- they will not run as regular Python programmes because they are not structured as so.
