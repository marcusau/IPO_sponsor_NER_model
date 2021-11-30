from tqdm import tqdm
import random
import re,csv,pickle, json,os,pathlib

def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and  tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

process_text_folder=r'C:\Users\marcus\PycharmProjects\ipo_pdf_scrapping\ML\NER\data\process'
process_text_filename=r'underwriting_iob2.txt'
process_text_file=open(os.path.join(process_text_folder,process_text_filename),'r',encoding='utf-8')

seq=process_text_file.read().split('\n\n')

iobs= [ [tuple(i.split('\t')) for i in line.split('\n')] for line  in seq]

for iob in iobs:
    try:
        word=[s[0] for s in iob]
        old_tags = [s[1] for s in iob]
        new_tags=iob_iobes(old_tags)
        print(word)
        print(old_tags)
        print(new_tags,'\n')


    except:
        print(iob)