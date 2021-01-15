#!/usr/bin/env python3
"""Parameters for the Aspect Based Hybrid Tweet Classifier.
"""
import os

DATABASE_FOLDER = ".{sep}data{sep}".format(sep = os.sep)
DATABASE_FILE   = "full-corpus.csv"
FILE_JAR        = "stanford-parser.jar"
FILE_MODELS_JAR = "stanford-parser-3.9.2-models.jar"
TEST_SIZE       = 0.20
RANDOM_STATE    = 40
