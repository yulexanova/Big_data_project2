from __future__ import print_function
import os
import sys
from pprint import pprint
from operator import add
import pyspark
from pyspark.context import SparkContext
import numpy as np
import scipy.sparse as sps
from pyspark.mllib.linalg import Vectors
import numpy
from math import log
import pandas

D = {}
sc = SparkContext()
file = "SampleData3.txt"

WordFrequencyDocs = sc.textFile(file, use_unicode=False)\
                    .map(lambda l: ((l.split(" ")[0], len([x for x in l.split(" ")[1:] if ("gene_" in x or "disease_" in x)])), [x for x in l.split(" ")[1:] if ("gene_" in x or "disease_" in x)]))\
                    .flatMap(lambda l: [((x, l[0]), 1) for x in l[1]]).cache()
WordFrequencyDocs = WordFrequencyDocs.reduceByKey(add).cache()
WordFrequencyDocs = WordFrequencyDocs.flatMap(lambda l: [((l[0][0]), (l[0][1][0], l[1], l[0][1][1]))]).cache()
WordCountCorpus = WordFrequencyDocs.countByKey()
WordFrequencyCorpus = WordFrequencyDocs.map(lambda l: [(l[0], l[1][0]), (l[1][1], l[1][2], WordCountCorpus[l[0]])]).cache()
WordFrequencyCorpus.cache()
Count = WordFrequencyCorpus.map(lambda l: l[0][1]).distinct().count()

# finished TFIDF by using spark
TFIDF = WordFrequencyCorpus.map(lambda l: [(l[0][0], l[0][1]), ((float(l[1][0])/float(l[1][1]))*(log(((Count)/(l[1][2])),10)))]).cache()

# start building vectors to compute Cos Similarity (term-term score)
DL = [x[0][1] for x in TFIDF.filter(lambda x: x[0][0]=='gene_nmdars_gene').collect()]
WL = [x for x in TFIDF.filter(lambda x: x[0][1] in DL).map(lambda x: x[0][0]).distinct().collect()]
DD = {x:TFIDF.filter(lambda l: l[0][0]==x).map(lambda x: (x[0][1].strip('doc'), x[1])).collect() for x in WL}

# Dictonary of term that are contained in at least one of the same documents as gene_nmdars_gene
pprint(DD)

#Potentially useable snippets for later on.
# LL = [(int(y.strip('doc')),z) for ((x,y),z) in nmdars.collect()]
# nmdars = TFIDF.filter(lambda l: l[0][0]=='gene_mhtt_gene')
# LL = [(int(y.strip('doc')),z) for ((x,y),z) in nmdars.collect()]
# print(sorted(LL))# VD = Vectors.dense(LL)
# print(VD)
# nmdarsDocs = nmdars.map(lambda l: l[0])
# TFIDF = TFIDF.map(lambda l: [(l[0][0]), [(l[0][1],l[1])]]).reduceByKey(add)
# TFIDF = TFIDF.filter(lambda l: (l[0]=="gene_nmdars_gene" or l[0]=="gene_ctr_gene"))
# TFIDF.map(lambda l: []*[]).sum()
# pprint(sorted(TFIDF.collect()))
# for x in sorted(TFIDF.collect()):
#     tmp = D.get(x[0], {})
#     for (y,z) in x[1]:
#         tmp[y] = z
#     D[x[0]] = tmp
# DF = pandas.DataFrame(D)
# tfq = DF.transpose().ix[:,['gene_calcitonin_gene','gene_ctr_gene']]
# print(tfq.dropna(thresh=1))

