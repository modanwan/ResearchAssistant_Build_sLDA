#!usr/bin/env
from utilities import *
import os
import random
import codecs
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer

class DataFormatter:
    """
        Format data from source
    """
    def __init__(self, temp_dir, stopswords_dir, businesswords_dir):
        """
        :param temp_dir: directory to store transit things
        :param stopswords_dir: directory of stopwords files, each file should be csv file and each line is a stopword
        :param businesswords_dir: similar to stopwords
        """
        self.stopwords = set(self.load_words(stopswords_dir))
        self.businesswords = set(self.load_words(businesswords_dir))

        self.temp_dir = temp_dir
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)

        self.temp_dir_rawcleaned = self.temp_dir + "/raw_cleaned"
        if not os.path.exists(self.temp_dir_rawcleaned):
            os.mkdir(self.temp_dir_rawcleaned)

        self.temp_dir_ngrams = self.temp_dir + "/ngrams"
        if not os.path.exists(self.temp_dir_ngrams):
            os.mkdir(self.temp_dir_ngrams)

        self.temp_dir_global_dic_path = os.path.join(self.temp_dir, "df.csv")
        self.temp_dir_vocab_dat = os.path.join(self.temp_dir, "vocab.dat")
        self.temp_mult_train_dat = os.path.join(self.temp_dir, "mult_train.dat")
        self.temp_mult_test_dat = os.path.join(self.temp_dir, "mult_test.dat")
        self.temp_depvars_train_adv_dat = os.path.join(self.temp_dir, "depvars_train_adv.dat")
        self.temp_depvars_test_adv_dat = os.path.join(self.temp_dir, "depvars_test_adv.dat")
        

    #load files,not used 
    #def load_files(self, doc_list, func=None):
        #print("loading files")
        #count = 0
        #for fn, fpath in doc_list:
            #count += 1
            #fpath_t = os.path.join(self.temp_dir_rawcleaned, fn+".txt")
            #with codecs.open(fpath, "r", "utf8", "ignore") as inf:
                #with open(fpath_t, "w") as outf:
                    #outf.write(func(inf.read()))
            #if count % 1000 == 0:
                #print(count)
        #print("successfully load files ...")

    @staticmethod
    def load_words(words_dir, ends=".csv"):
        flist = read_dir(words_dir, ends=ends)
        wordslist = []
        for _, fpath in flist:
            with open(fpath, "r") as f:
                for word in f.readlines():
                    wordslist.append(word.strip())
        return wordslist

    def generate_all(self, train_ratio):

        # load surprise
        surprise_dic = load_surprise()
        surprise_dic_adv = load_surprise()
       

        # prepare training set and test set
        flist = read_dir(self.temp_dir_ngrams, ends=".txt")
        flist = [f for f in flist if f[0] in surprise_dic]
        train_len = int(train_ratio * len(flist))
        train_set = set(random.sample(flist, train_len))
        test_set = set(flist) - train_set
        train_list = sorted(list(train_set))
        test_list = sorted(list(test_set))

        # load doc frequency
        with open(self.temp_dir_global_dic_path, "r") as f:
            wordslist = f.readlines()
        df = {}
        for w in wordslist:
            w = w.strip().split(",")
            df[w[0]] = w[1]

        # build vocab using training set
        vocab = set()
        for file_name, file_path in train_set:
            vocab_adoc = load_csvdic(file_path)
            for k, _ in vocab_adoc.items():
                if (k in df) and int(df[k]) >= 100 and int(df[k]) <5983:
                    vocab.add(k)

        vocab_list = sorted(list(vocab))
        vocab_indexmap = dict([(vocab_list[i], i) for i in range(len(vocab_list))])

        # output vocab.dat
        with open(self.temp_dir_vocab_dat, "w") as wf:
            for w in vocab_list:
                wf.write(w+"\n")

        # output mult_train.dat
        self._write_data(self.temp_mult_train_dat, train_list, vocab_indexmap)
        # output depvars_train_adv.dat
        self._write_label(self.temp_depvars_train_adv_dat, train_list, surprise_dic_adv)
       
        # output mult_test.dat
        self._write_data(self.temp_mult_test_dat, test_list, vocab_indexmap)
        # output depvars_test_adv.dat
        self._write_label(self.temp_depvars_test_adv_dat, test_list, surprise_dic_adv)
       
    @staticmethod
    def _write_data(target_path, doc_list, vocab_indexmap):
        with open(target_path, "w") as wf:
            for doc_name, doc_path in doc_list:
                wordcounts = load_csvdic(doc_path)
                wordcounts_by_index = []
                for word, value in wordcounts.items():
                    if word not in vocab_indexmap:
                        continue
                    wordcounts_by_index.append((vocab_indexmap[word], value))
                wordcounts_by_index = sorted(wordcounts_by_index)
                wf.write(str(len(wordcounts_by_index)))
                for word_index, value in wordcounts_by_index:
                    wf.write(" " + str(word_index) + ":" + str(value))
                wf.write("\n")

    @staticmethod
    def _write_label(target_path, doc_list, surprise_dic):
        with open(target_path, "w") as wf:
            for doc_name, _ in doc_list:
                wf.write("%s|%s\n" % (doc_name, str(surprise_dic[doc_name])))

##### does not be used 
    #def generate_grams(self):
        #print("generating grams")
        #count = 0
        #flist = read_dir(self.temp_dir_rawcleaned, ends=".txt")
        #global_words_dic = {}
        #for fn, fpath in flist:
            #count += 1
            #if count % 1000 == 0:
                #print(count)
            #fpath_t = os.path.join(self.temp_dir_ngrams, fn)
            #with open(fpath, "r") as f:
                #grams_dic = self._generate_grams_doc(f.read())
            #with open(fpath_t, "w") as f:
                #for k, val in grams_dic.items():
                    #s = k + "," + str(val) + '\n'
                    #f.write(s)
                    #global_words_dic[k] = global_words_dic.get(k, 0) + 1
        #with open(self.temp_dir_global_dic_path, "w") as f:
            #for k, val in global_words_dic.items():
                #if val < 100:
                    #continue
                #f.write(k + "," + str(val) + "\n")
        #print("successfully generate grams ...")

    def generate_grams_from_raw_doc(self, doc_list, func=None):
        print("generating grams")
        count = 0
        global_words_dic = {}
        for fn, fpath in doc_list:
            count += 1
            if count % 1000 == 0:
                print(count)
            fpath_t = os.path.join(self.temp_dir_ngrams, fn + ".txt")

            with codecs.open(fpath, "r", "utf8", "ignore") as inf:
                grams_dic = self._generate_grams_doc_with_lemmatize(func(inf.read()))

            with open(fpath_t, "w") as f:
                for k, val in grams_dic.items():
                    s = k + "," + str(val) + '\n'
                    f.write(s)
                    global_words_dic[k] = global_words_dic.get(k, 0) + 1

        with open(self.temp_dir_global_dic_path, "w") as f:
            #common=["thousand","euro","million","billion","year","company","Fiscal","financial","consolid"]
            for k, val in global_words_dic.items():
                if (val < 100) or (val >=5983):
                    continue
                f.write(k + "," + str(val) + "\n")

        print("successfully generate grams ...")



    # generate grams without lemmatize
    def _generate_grams_doc(self, content):
        lines = content.split("\n")
        gram_dic = {}
        for line in lines:
            words = [w.strip() for w in line.strip().split(" ")]
            one_gram = words
            two_gram = zip(one_gram, one_gram[1:])

            for w in one_gram:
                if (w not in self.stopwords) and (w in self.businesswords):
                    gram_dic[w] = gram_dic.get(w, 0) + 1

            for a, b in two_gram:
                if (
                    (a not in self.stopwords) and
                    (b not in self.stopwords) and
                    (a in self.businesswords) and
                    (b in self.businesswords)
                ):
                    k = a + "---" + b
                    gram_dic[k] = gram_dic.get(k, 0) + 1
        return gram_dic

    # generate grams with lemmatize
    def _generate_grams_doc_with_lemmatize(self, content):
        lines = content.split("\n")
        gram_dic = {}
        common1=["fin"]
        common2=["thousand","euro","mil","bil","year","company","fisc","consolid"]
        for line in lines:
            words = [w.strip() for w in line.strip().split(" ")]
            one_gram = words
            two_gram = zip(one_gram, one_gram[1:])

            for w in one_gram:
                if (w not in self.stopwords) and (w in self.businesswords) and (w not in common1) and (self._porter(w) not in common2):
                    gram_dic[self._lemmatize(w)] = gram_dic.get(self._lemmatize(w), 0) + 1

            for a, b in two_gram:
                if (
                    (a not in self.stopwords) and
                    (b not in self.stopwords) and
                    (a in self.businesswords) and
                    (b in self.businesswords) and
                    (self._porter(a) not in common2) and
                    (self._porter(b) not in common2)
                ):
                    k = self._lemmatize(a) + "-" + self._lemmatize(b)
                    gram_dic[k] = gram_dic.get(k, 0) + 1
        return gram_dic
     
        # generate grams with porter
    def _generate_grams_doc_with_porter(self, content):
        lines = content.split("\n")
        gram_dic = {}
        common1=["financial"]
        common2=["thousand","euro","million","billion","year","compani","fiscal","consolid"]
        for line in lines:
            words = [w.strip() for w in line.strip().split(" ")]
            one_gram = words
            two_gram = zip(one_gram, one_gram[1:])
            
            for w in one_gram:
                if (w not in self.stopwords) and (w in self.businesswords) and (w not in common1) and (self._porter(w) not in common2):
                    gram_dic[self._porter(w)] = gram_dic.get(self._porter(w), 0) + 1

            for a, b in two_gram:
                if (
                    (a not in self.stopwords) and
                    (b not in self.stopwords) and
                    (a in self.businesswords) and
                    (b in self.businesswords) and
                    (self._porter(a) not in common2) and
                    (self._porter(b) not in common2)
                ):
                    k = self._porter(a) + "-" + self._porter(b)
                    gram_dic[k] = gram_dic.get(k, 0) + 1
        return gram_dic

    def _lemmatize(self, w):
        wnl = WordNetLemmatizer()
        lancaster_stemmer = LancasterStemmer()
        pos = ['a', 's', 'r', 'n', 'v']
        for p in pos:
            nw = wnl.lemmatize(w, p)
            if nw != w:
                return lancaster_stemmer.stem(nw)
        return lancaster_stemmer.stem(w)
    
    def _porter(self, w):
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        return ps.stem(w)