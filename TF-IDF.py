from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pprint import pprint
import pandas

sc = SparkContext()
file = "SampleData.txt"
termSet = set()
hashVals = {}
D = {}
docs = []

documents = sc.textFile(file).map(lambda line: line.split(" ")[1:])

with open(file, 'r') as f:
    for line in f:
        docs.append(line.split(" ")[0].strip(' '))

hashingTF = HashingTF()

for x in documents.collect():
    for y in x:
        termSet.add(y)

tf = hashingTF.transform(documents)
tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

for x in termSet:
    hashVals[hashingTF.indexOf(str(x))] = str(x)

for x, y in zip(docs, tfidf.collect()):
    for w, z in zip([hashVals[u] for u in y.indices], y.values):
        tmp = D.get(x, {})
        tmp[w] = z
        D[x] = tmp

DF = pandas.DataFrame(D)
sc.stop()
print(DF.transpose())
