from tqdm import tqdm
import random
import re,csv,pickle, json,os,pathlib
from cleantext import clean

def sliding_window_list(sentence:str,size:int=3):
    sentence = sentence.split(' ')

    for i in range(len(sentence) - size + 1):
        batch = sentence[i:i + size]
        yield i,i + size,batch


raw_text_folder=r'C:\Users\marcus\PycharmProjects\ipo_pdf_scrapping\ML\NER\data\raw'
raw_text_filename=r'underwriting.txt'
raw_text_file=os.path.join(raw_text_folder,raw_text_filename)

process_text_folder=r'C:\Users\marcus\PycharmProjects\ipo_pdf_scrapping\ML\NER\data\process'
process_text_filename=r'underwriting_iob.txt'
process_text_file=open(os.path.join(process_text_folder,process_text_filename),'w',encoding='utf-8')


with open(raw_text_file,'r',encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        line=[clean(l.strip(),normalize_whitespace=True,lower=False) for l in line.strip().split('\t') ]
        line=[l for l in line if l !='']
        text,labels=line[0],line[1:]

        word = labels[0::2]
        tag=labels[1::2]
        annotations=[(w,t) for w,t in zip(word,tag)]


        textlist=text.split(' ')
        textlist_len=len(textlist)
        textlist_spans=[i for i in range(0,len(textlist))]

        annotation_spans={}
        for annotation in annotations:
            search_words, tag = annotation[0], annotation[1]
            search_wordlist=search_words.split(' ')
            for i in sliding_window_list(sentence=text,size=len(search_wordlist)):
                sub_start,sub_end,sub_wordlist=i[0],i[1],i[2]
                if sub_wordlist==search_wordlist:
                    len_search_wordlist=len(search_wordlist)
                    if len_search_wordlist==1:
                        search_word_tags=[f'S-{tag}']
                    elif len_search_wordlist==2:
                        search_word_tags = [f'B-{tag}']+[f'E-{tag}']

                    elif len_search_wordlist >2:
                        search_word_tags = [f'B-{tag}'] + [f'I-{tag}']*(len_search_wordlist-2)+[f'E-{tag}']

                    search_word_spans=[j for j in range(sub_start,sub_end)]

                    for search_span, search_word_tag in zip(search_word_spans,search_word_tags):
                        annotation_spans[search_span]= search_word_tag

                    break

        text_0_span=d1=filter(lambda x:x not in annotation_spans.keys() ,textlist_spans)
        text_0_span={s:'O' for s in text_0_span}

        text_spans=annotation_spans.copy()
        text_spans.update(text_0_span)

        for w,s in zip(textlist,text_spans.values()):
            process_text_file.write(w+'\t'+s+'\n')
        process_text_file.write('\n')

process_text_file.close()

