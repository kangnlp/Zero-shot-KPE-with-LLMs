import re
import codecs
import json
import os
import nltk
from tqdm import tqdm
import argparse


def clean_text(text="",database="Inspec"):

    #Specially for Duc2001 Database
    if(database=="Duc2001" or database=="Semeval2017"):
        pattern2 = re.compile(r'[\s,]' + '[\n]{1}')
        while (True):
            if (pattern2.search(text) is not None):
                position = pattern2.search(text)
                start = position.start()
                end = position.end()
                # start = int(position[0])
                text_new = text[:start] + "\n" + text[start + 2:]
                text = text_new
            else:
                break

    pattern2 = re.compile(r'[a-zA-Z0-9,\s]' + '[\n]{1}')
    while (True):
        if (pattern2.search(text) is not None):
            position = pattern2.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + " " + text[start + 2:]
            text = text_new
        else:
            break

    pattern3 = re.compile(r'\s{2,}')
    while (True):
        if (pattern3.search(text) is not None):
            position = pattern3.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[:start + 1] + "" + text[start + 2:]
            text = text_new
        else:
            break

    pattern1 = re.compile(r'[<>[\]{}]')
    text = pattern1.sub(' ', text)
    text = text.replace("\t", " ")
    text = text.replace(' p ','\n')
    text = text.replace(' /p \n','\n')
    lines = text.splitlines()
    # delete blank line
    text_new=""
    for line in lines:
        if(line!='\n'):
            text_new+=line+'\n'

    return text_new

def get_long_data(file_path="data/nus/nus_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                fulltxt = jsonl['fulltext']
                doc = ' '.join([abstract, fulltxt])
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="nus")
                doc = doc.replace('\n', ' ')
                data[jsonl['name']] = doc
                labels[jsonl['name']] = keywords
            except:
                raise ValueError
    return data,labels

def get_short_data(file_path="data/kp20k/kp20k_valid2k_test.json"):
    """ Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(file_path, 'r', 'utf-8') as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl['keywords'].lower().split(";")
                abstract = jsonl['abstract']
                doc =abstract
                doc = re.sub('\. ', ' . ', doc)
                doc = re.sub(', ', ' , ', doc)

                doc = clean_text(doc, database="kp20k")
                doc = doc.replace('\n', ' ')
                data[i] = doc
                labels[i] = keywords
            except:
                raise ValueError
    return data,labels


def get_duc2001_data(file_path="data/DUC2001"):
    pattern = re.compile(r'<TEXT>(.*?)</TEXT>', re.S)
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            if (fname == "annotations.txt"):
                # left, right = fname.split('.')
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                lines = text.splitlines()
                for line in lines:
                    left, right = line.split("@")
                    d = right.split(";")[:-1]
                    l = left
                    labels[l] = d
                f.close()
            else:
                infile = os.path.join(dirname, fname)
                f = open(infile,'rb')
                text = f.read().decode('utf8')
                text = re.findall(pattern, text)[0]

                text = text.lower()
                text = clean_text(text,database="Duc2001")
                data[fname]=text.strip("\n")
                # data[fname] = text
    return data,labels

def get_inspec_data(file_path="data/Inspec"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(file_path):
        for fname in filenames:
            left, right = fname.split('.')
            if (right == "abstr"):
                infile = os.path.join(dirname, fname)
                f=open(infile)
                text=f.read()
                text = text.replace("%", '')
                text=clean_text(text)
                data[left]=text
            if (right == "uncontr"):
                infile = os.path.join(dirname, fname)
                f=open(infile)
                text=f.read()
                text=text.replace("\n",' ')
                text=clean_text(text,database="Inspec")
                text=text.lower()
                label=text.split("; ")
                labels[left]=label
    return data,labels

def get_semeval2017_data(data_path="data/SemEval2017/docsutf8",labels_path="data/SemEval2017/keys"):

    data={}
    labels={}
    for dirname, dirnames, filenames in os.walk(data_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            # f = open(infile, 'rb')
            # text = f.read().decode('utf8')
            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", '')
            text = clean_text(text,database="Semeval2017")
            data[left] = text.lower()
            # f.close()
    for dirname, dirnames, filenames in os.walk(labels_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            f = open(infile, 'rb')
            text = f.read().decode('utf8')
            text = text.strip()
            ls=text.splitlines()
            labels[left] = ls
            f.close()
    return data,labels


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data', help="Directory path of test datasets")
    parser.add_argument('--max_len', type=str, default='512', help="Max length of input document")
    args = parser.parse_args()

    data_path = args.data_path
    MAX_LEN = int(args.max_len)

    dataset_list = ['Inspec', 'SemEval2017', 'SemEval2010', 'DUC2001', 'nus', 'krapivin']

    for dataset_name in dataset_list:

        dataset_dir = os.path.join(data_path, dataset_name)

        if dataset_name =="SemEval2017":
            data, referneces = get_semeval2017_data(dataset_dir + "/docsutf8", dataset_dir + "/keys")
        elif dataset_name == "DUC2001":
            data, referneces = get_duc2001_data(dataset_dir)
        elif dataset_name == "nus" :
            data, referneces = get_long_data(dataset_dir + "/nus_test.json")
        elif dataset_name == "krapivin":
            data, referneces = get_long_data(dataset_dir + "/krapivin_test.json")
        elif dataset_name == "kp20k":
            data, referneces = get_short_data(dataset_dir + "/kp20k_valid200_test.json")
        elif dataset_name == "SemEval2010":
            data, referneces = get_short_data(dataset_dir + "/semeval_test.json")
        elif dataset_name == "Inspec":
            data, referneces = get_inspec_data(dataset_dir)


        docs = []
        labels = []
        labels_stemmed = []

        porter = nltk.PorterStemmer()

        for key, doc in data.items():

            # Get stemmed labels and document segments
            labels.append([ref.replace(" \n", "") for ref in referneces[key]])
            labels_s = []
            for l in referneces[key]:
                tokens = l.split()
                labels_s.append(' '.join(porter.stem(t) for t in tokens))

            doc = ' '.join(doc.split()[:MAX_LEN])
            
              
            labels_stemmed.append(labels_s)
            docs.append(doc)
        
        assert len(docs) == len(labels) == len(labels_stemmed), "The lengths of doc_list, labels, and labels_stemed are not equal."
        
        jsonl_lines = []
        for doc, label, stemmed_label in zip(docs, labels, labels_stemmed):
            line = {}
            line['doc'] = doc
            line['label'] = label
            line['stemmed_label'] = stemmed_label
            jsonl_lines.append(line)

        result_path = os.path.join(data_path, 'processed')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print(f"Directory created: {result_path}")

        file_path = os.path.join(result_path, f'{dataset_name}_MAX{MAX_LEN}.jsonl')

        with open(file_path, "w", encoding='utf-8') as f:
            for json_data in jsonl_lines:
                f.write(json.dumps(json_data, ensure_ascii=False)+'\n')
