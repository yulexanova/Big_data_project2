# Marcello Martins & Yulia Emelyanova Big Data Project 2

from __future__ import print_function
from pprint import pprint
from operator import add
from pyspark.context import SparkContext, SparkConf
from math import log, sqrt
import os
import sys

# R <-- RAW INPUT list of TERMS seperated by spaces
# t <-- Individual TERM
# d <-- Individual DOCUMENT
# N <-- Total number of TERMS in a given DOCUMENT
# M <-- Number of times a given TERM appears in a given DOCUMENT
# T <-- Total number of documents where a given TERM apperars
# D <-- Total number of unique documents in the corpus

# TFIDF <-- Computed TD-IDF for a given TERM and DOCUMENT
# QTD   <-- Dictonary containing TFIDFs for DOCUMENTs based on the QUERYTERM
# RT    <-- List containing all the TERMS which are related to the QUERYTERM
# CSD   <-- Dictonary containing denominators for the cos similarity formula

# QUERYTERM <-- The Term compared against when determing cos similairty
# RESULTS   <-- The sorted results list for SIMILARITY based on the QUERYTERM
# SIMILARITY<-- The Computed cos similarity for the pair (QUERYTERM,t)


def main(sc, filename, queryterm):
    FILENAME = sc.broadcast(filename)
    QUERYTERM = sc.broadcast(queryterm)

    # Take Raw Input file and map each line into the form of ((T,(d,N)),1)
    # (d R) --> ((t,(d,N)),1)
    TermFrequencyDocs = sc.textFile(FILENAME.value, use_unicode=False)\
        .map(
                lambda line:
                (
                    (
                        line.split(" ")[0],
                        len(
                            [x for x in line.split(" ")[1:]
                                if ("gene_" in x or "disease_" in x)]
                           )
                    ),
                    [x for x in line.split(" ")[1:] if
                        ("gene_" in x or "disease_" in x)]
                )
             )\
        .flatMap(lambda line: [((x, line[0]), 1) for x in line[1]])

    # Reduce by key by adding up each corresponding line
    # ((t,(d,N)),1) --> ((t,(d,N)),M)
    TermFrequencyDocs = TermFrequencyDocs\
        .reduceByKey(add)

    # Map each line into the form of (t,(d,N,M))
    # ((t,(d,N)),M) --> (t,(d,M,N))
    TermFrequencyDocs = TermFrequencyDocs\
        .map(
                lambda line:
                (
                    (line[0][0]),
                    (line[0][1][0], line[1], line[0][1][1])
                )
            )

    # Count each line by key to get the total number of documents each term
    # appears Store the result in a broadcasted Dictonary
    TermCountCorpus = sc.broadcast(TermFrequencyDocs.countByKey())

    # Map each line into the form ((t,d),(M,N,T))
    # (t,(d,M,N)) --> ((t,d),(M,N,T))
    TermFrequencyCorpus = TermFrequencyDocs\
        .map(
                lambda line:
                [
                    (line[0], line[1][0]),
                    (line[1][1], line[1][2], TermCountCorpus.value[line[0]])
                ]
            )

    # Count the total number of unique documents in the corpus and broadcast it
    Count = sc.broadcast(
        TermFrequencyCorpus
        .map(
                lambda line:
                line[0][1]
            ).distinct().count()
    )

    # Compute TF-IDF for each line mapping to the form of ((t,d),TFIDF)
    # ((t,d),(M,N,T)) --> ((t,d),TFIDF)
    TFIDF = TermFrequencyCorpus\
        .map(
                lambda l:
                [
                    (l[0][0], l[0][1]),
                    ((float(l[1][0])/float(l[1][1])) *
                     (log(((Count.value)/(l[1][2])))))
                ]
            )

    # Create a Boradcasted Dictonary containing all the TFIDFs for each
    # DOCUMENT pertaining to the given QUERYTERM.
    QTD = sc.broadcast(
        {
            line[0][1]: line[1] for line in (
                TFIDF.filter(
                    lambda line:
                        (line[0][0] == QUERYTERM.value)
                ).collect()
            )
        }
    )

    # Create a list which contains all the TERMS which are related to the
    # QUERYTERM by having sharing at least one document in common.
    RT = sc.broadcast(
        TFIDF.filter(
            lambda line:
            (line[0][1] in QTD.value)
        ).map(
            lambda line:
                line[0][0]
        ).distinct().collect()
    )

    # Create a dictonary containing the square root of the summation of the
    # squared TFIDFs keyed by a given TERM.
    CSD = sc.broadcast(
        {
            line[0]: line[1] for line in TFIDF.filter(
                lambda line:
                (
                    line[0][0] in RT.value
                )
            ).map(
                lambda line:
                (
                    (line[0][0]),
                    (line[1]*line[1]))
                )
            .reduceByKey(add).map(
                lambda line:
                (
                    (line[0]),
                    (sqrt(line[1]))
                )
            ).collect()
        }
    )

    # Compute the cos similairty for a given QUERYTERM and it's related terms
    # by computing the summation of TFIDFs for the QUERYTERM and it's related
    # terms who share DOCUMENTs,divided by the the product of the two values
    # returned by CSD[A] * CSD[B], then sort the final results into decending
    # order in the form of ((QUERYTERM,t), SIMILARITY)
    RESULTS = TFIDF.filter(
        lambda line:
        (
            line[0][0] != QUERYTERM.value and line[0][1] in QTD.value
        )
    ).map(
        lambda line:
        (
            (QUERYTERM.value, line[0][0]),
            (QTD.value[line[0][1]]*line[1]))
        ).reduceByKey(add).map(
            lambda line:
            (
                (line[0][0], line[0][1]),
                (
                    (line[1]) /
                    (CSD.value[line[0][0]] *
                        CSD.value[line[0][1]])
                )
                )
            ).sortBy(
                lambda line: line[1],
                ascending=False
            )

    # print the results to screen and also write to file results.txt
    print("Query term: {}".format(QUERYTERM.value))
    for line in RESULTS.collect():
        print("{}{}".format((line[0][1]+':').ljust(60), line[1]))
    with open('results.txt', 'w') as f:
        f.write("Query term: {}\n".format(QUERYTERM.value))
        for line in RESULTS.collect():
            f.write("{}{}\n".format((line[0][1]+':').ljust(60), line[1]))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage: $SPARK_HOME/bin/spark-submit Project2.py " +
              "<HostUrl> <InputFile> <QueryTerm>\n" +
              "If running on local, replace <HostUrl> with \"local\"\n" +
              "Please ensure that you have Spark installed and" +
              " can resolve $SPARK_HOME to the correct directory.\n")
        sys.exit()
    conf = (SparkConf()
            .setMaster(sys.argv[1])
            .setAppName("Project2"))
    sc = SparkContext(conf=conf)
    filename = sys.argv[2]
    queryterm = sys.argv[3]
    main(sc, filename, queryterm)
