import os
import sys
from pprint import pprint
from operator import add
import pyspark
from pyspark.context import SparkContext

sc = SparkContext()
file = "SampleData3.txt"

wordcounts = sc.textFile(file) \
        .map(lambda l: ((l.split(" ")[0], len([x for x in l.split(" ")[1:] if ("gene_" in x or "disease_" in x)])), [x for x in l.split(" ")[1:] if ("gene_" in x or "disease_" in x)]))\ \
        .flatMap(lambda x: x.split()) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x,y:x+y) \
        .map(lambda x:(x[1],x[0])) \
        .sortByKey(False) 


