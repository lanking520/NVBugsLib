#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import logging
import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
import os.path
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import sys

logger = logging.getLogger(__name__)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '='):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '.' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()

class SentenceParser:

    regex_str = [
        r'<[^>]+>',                                                                     # HTML tags
        r'(?:@[\w_]+)',                                                                 # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",                                               # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',   # URLs
        r'(?:(?:\d+,?)+(?:\.?\d+)?)',                                                   # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])",                                                    # words with - and '
        r'(?:[\w_]+)',                                                                  # other words
        r'(?:\S)'                                                                       # anything else
    ]

    def __init__(self, loggingLevel = 20):
        self.data = None
        self.porter = nltk.PorterStemmer()
        self.stops = set(stopwords.words("english"))
        logging.basicConfig(level=loggingLevel)
        pass

    # Read File and Import File Section
    def read_file(self, filepath, filetype, encod ='ISO-8859-1', header =None):
        logger.info('Start reading File')
        if not os.path.isfile(filepath):
            logger.error("File Not Exist!")
            sys.exit()
        if filetype == 'csv':
            df = pd.read_csv(filepath, encoding=encod, header =header)
        elif filetype == 'json':
            df = pd.read_json(filepath, encoding=encod, lines=True)
        elif filetype == 'xlsx':
            df = pd.read_excel(filepath, encoding=encod, header =header)
        else:
            logger.error("Extension Type not Accepted!")
            sys.exit()

        logger.debug(df)
        self.data = df

    def import_data(self, data):
        logger.info('Import DataFrame')
        if isinstance(data, pd.core.frame.DataFrame):
            self.data = data
        else:
            logger.error("Data Type not Accepted! Please use pandas.core.frame.DataFrame")
            sys.exit()

    # Operation on the Dataset 
    def merge_column(self, columns, name):
        logger.info('Merge headers %s to %s', str(columns), name)
        self.data[name] = ''
        for header in columns:
            self.data[name] += ' ' + self.data[header]

    def split_by_column(self,column, reset_index = False):
        logger.info("Start Spliting data through the column values")
        mylist = self.data[column].unique()
        result = {}
        printProgressBar(0, mylist.shape[0], prefix='Progress:', suffix='Complete', length=50)
        idx =0
        for row in mylist:
            if reset_index:
                result[row] = self.data.loc[self.data[column] == row].reset_index(drop = True)
            else:
                result[row] = self.data.loc[self.data[column] == row]
            printProgressBar(idx+1, mylist.shape[0], prefix='Progress:', suffix='Complete', length=50)
            # print("\nThe Shape of "+ row + " is "+str(result[row].shape))
            idx += 1
        return result

    def get_all_headers(self):
        return list(self.data.columns.values)

    def get_column(self,column):
        return self.data[column].values.tolist()

    # Start cleaning the text from column
    def process_text(self, column, remove_symbol = True, remove_stopwords=False, stemming=False):
        logger.info("Start Data Cleaning...")
        self.data[column] = self.data[column].str.replace(r'[\n\r\t]+', ' ')
        # Remove URLs
        self.data[column] = self.data[column].str.replace(self.regex_str[3],' ')
        tempcol = self.data[column].values.tolist()
        
        # This part takes a lot of times
        printProgressBar(0, len(tempcol), prefix='Progress:', suffix='Complete', length=50)
        for i in range(len(tempcol)):
            row = BeautifulSoup(tempcol[i],'html.parser').get_text()
            if remove_symbol:
                row = re.sub('[^a-zA-Z0-9]', ' ', row)
            words = row.split()
            if remove_stopwords:
                words = [w for w in words if not w in self.stops and not w.replace('.', '', 1).isdigit()]
            if stemming:
                words = [self.porter.stem(w) for w in words]
            row = ' '.join(words)
            tempcol[i] = row.lower()
            printProgressBar(i+1, len(tempcol), prefix='Progress:', suffix='Complete', length=50)
        print("\n")
        return tempcol

    # Cleaning text based on text input
    def process_line(self, line, remove_symbol = True, remove_stopwords=False, stemming = False):
        line.replace(r'[\n\r\t]+', ' ')
        line.replace(self.regex_str[3],' ')
        row = BeautifulSoup(line,'html.parser').get_text()
        if remove_symbol:
            row = re.sub('[^a-zA-Z0-9]', ' ', row)
        words = row.split()
        if remove_stopwords:
            words = [w for w in words if not w in self.stops and not w.replace('.', '', 1).isdigit()]
        if stemming:
                words = [self.porter.stem(w) for w in words]
        row = ' '.join(words)
        return row

    # Designed for NVBugs for Synopsis Clean
    def synop_clean(self, column , new_name):
        logger.info("Synopsis Cleaner Start...")
        df = self.data
        df[new_name] = df[column].str.replace(r'[\n\r\t]+', ' ')
        df[new_name] = df[column].str.replace(r'[\[\]/:?\\"|]+', ' ')

    # helper function for Description Clean
    def matchhtml(self, test_str):
        pattern = r'http[s]?://(?:[a-z]|[0-9]|[=#$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
        return re.findall(pattern, test_str), re.sub(pattern, ' ', test_str)

    def matchfile(self, test_str):
        pattern = r'(( [A-Za-z]:|\\|/|\./)[a-zA-Z0-9_\.\\/#&~!%]+)'
        templist = []
        for item in re.findall(pattern, test_str):
            templist.append(item[0])
        return templist, re.sub(pattern, ' ', test_str)

    def removequot(self, test_str):
        pattern = r'&.*?;'
        text = re.sub(pattern, ' ', test_str)
        text = re.sub(r'[\[\]/:?\\"|`,]+', ' ',text)
        return text

    def removebracket(self, test_str):
        pattern = r'<.*?>'
        return re.sub(pattern, ' ', test_str)

    # Designed for NVBugs for Description Clean
    def description_clean(self, column, new_name):
        logger.info("Description Cleaner Start...")
        df = self.data
        df[new_name] = df[column].str.replace(r'[\n\r\t\a\b]+', ' ')
        # Remove non-printable words
        df[new_name] = df[new_name].str.replace(r'[^\x00-\x7f]+', '')
        # df[new_name] = df[column].str.replace(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',' ')
        templist = df[new_name].tolist()
        htmls = []
        converted = []
        filepaths = []
        printProgressBar(0, df.shape[0], prefix = 'Progress', suffix = 'Completed',length = 50)
        for idx, row in enumerate(templist):
            # Determine if it is float('nan')
            if row == row:
                html, str_nohtml = self.matchhtml(row)
                htmls.append(' '.join(html))
                filepath, str_nofilepath = self.matchfile(str_nohtml)
                filepaths.append(' '.join(filepath))
                text = BeautifulSoup(str_nofilepath,'html.parser').get_text()
                text = self.removebracket(text)
                text = self.removequot(text)
                converted.append(text)
            else:
                converted.append(row)
                htmls.append('')
                filepaths.append('')
                # converted.append(row)
            printProgressBar(idx, df.shape[0], prefix = 'Progress', suffix = 'Completed',length = 50)
        df[new_name] = converted
        df['html'] = htmls
        df['filepath'] = filepaths

    # Statistic area, create vectorizer for analysis
    def create_vectorizer(self, text, max_features = 1000, n_gram=(1,4)):
        logger.info("Creating Counting Vectorizer...")
        self.vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             ngram_range= n_gram,
                             max_features = max_features)

        data_vector = self.vectorizer.fit_transform(text)
        data_vector = data_vector.toarray()
        vocab = self.vectorizer.get_feature_names()
        self.data_df = pd.DataFrame(data=data_vector, columns=vocab)
        return self.data_df

    def get_top(self):
        return self.data_df.sum().sort_values(ascending=False)

    def get_sample(self, num):
        return self.data.sample(num)
