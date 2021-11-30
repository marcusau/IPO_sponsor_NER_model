
from tqdm import tqdm
import random
from difflib import SequenceMatcher
import re,csv,pickle, json,os
from cleantext import clean

def matcher(string, pattern):

    match_list = []
    pattern = pattern.strip()
    seqMatch = SequenceMatcher(None, string, pattern, autojunk=False)
    match = seqMatch.find_longest_match(0, len(string), 0, len(pattern))
    if (match.size == len(pattern)):
        start = match.a
        end = match.a + match.size
        match_tup = (start, end)
        string = string.replace(pattern, "X" * len(pattern), 1)
        match_list.append(match_tup)

    return match_list, string


def mark_sentence(s, match_list):

    word_dict = {}
    for word in s.split():
        word_dict[word] = 'O'

    for start, end, e_type in match_list:
        temp_str = s[start:end]
        tmp_list = temp_str.split()
        if len(tmp_list) > 1:
            word_dict[tmp_list[0]] = 'B-' + e_type
            for w in tmp_list[1:]:
                word_dict[w] = 'I-' + e_type
        else:
            word_dict[temp_str] = 'B-' + e_type
    return word_dict



def create_data(texts, annotations, filepath=None):


    f= open(filepath , 'w',encoding='utf-8')


    for text, annotation in zip(texts, annotations):
        match_list = []
        try:

            for i in annotation:
                a, text_ = matcher(text, i[0])
                match_list.append((a[0][0], a[0][1], i[1]))
        except Exception as e:
            pass
        d = mark_sentence(text, match_list)

        for i in d.keys():
            f.writelines(i + '\t' + d[i] +'\n')
        f.writelines('\n')
    f.close()

if __name__=='__main__':
    output=r'C:\Users\marcus\PycharmProjects\ipo_pdf_scrapping\ML\NER\data\process\train.txt'

    with open(r'C:\Users\marcus\PycharmProjects\ipo_pdf_scrapping\ML\NER\data\semi_process.txt','r',encoding='utf-8') as f:
        rows = csv.reader(f, delimiter='\t')

        texts,annotations=[],[]
        for row in tqdm(rows):
            text=row[0]
            labels=[r for r in row[1:] if r is not '']
            annotation=[(clean(word,lower=False,fix_unicode=True,strip_lines=True,normalize_whitespace=True).replace('  ',' '),tag) for word,tag in zip(labels[::2],labels[1::2])]
            texts.append(text)
            annotations.append(annotation)

    create_data(texts, annotations, filepath=output)

    with open(output,'r',encoding='utf-8') as f:
        lines=  f.read().split('\n\n')
        lines=[['\t'.join([t.strip() for t in l.split('\t')]) for l in line.split('\n')] for line in lines]
        lines=sorted(lines,key=len,reverse=True)
        for line in lines[:10]:
            print(len(line),line)
