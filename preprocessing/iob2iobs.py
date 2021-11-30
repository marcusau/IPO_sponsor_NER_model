import os,re

g=open(r'C:\Users\marcus\PycharmProjects\ipo_pdf_scrapping\ML\NER\data\process\revised_full.txt','a',encoding='utf-8')
with open(r'C:\Users\marcus\PycharmProjects\ipo_pdf_scrapping\ML\NER\data\process\full.txt','r',encoding='utf-8') as f:
    ff=f.read()
    for line in ff.split('\n\n'):
        line=[l.split('\t') for l in line.split('\n')]
        words=[i[0] for i in line if len(i)==2]
        tags=[i[1] for i in line  if len(i)==2]
        heads,types=[],[]
        for tag in tags:
            if tag != 'O':
                head,nertype=tag.split('-')
            else:
                head, nertype ='O','O'
            heads.append(head)
            types.append(nertype)


        heads=''.join(heads)
        if heads.endswith('BI'):
            heads=heads[:-2]+'BE'
        if heads.endswith('BII'):
            heads=heads[:-3]+'BIE'
        if heads.endswith('BIII'):
            heads=heads[:-4]+'BIIE'
        if heads.endswith('BIIII'):
            heads=heads[:-5]+'BIIIE'
        if heads.endswith('BIIIII'):
            heads=heads[:-6]+'BIIIIE'
        if heads.endswith('BIIIIII'):
            heads=heads[:-7]+'BIIIIIE'
        if heads.endswith('BIIIIIII'):
            heads=heads[:-8]+'BIIIIIIE'
        if heads.endswith('BIIIIIIII'):
            heads=heads[:-9]+'BIIIIIIE'


        if heads.endswith('OI'):
            heads=heads[:-2]+'OO'
        heads=re.sub('IO','EO',heads)
        heads=re.sub('IB','EB',heads)
        heads=re.sub('OBO','OSO',heads)
        heads=re.sub('BIO','BIE',heads)


        if len(words)==1 and heads!='O':
            heads='S'

        if len(words)==2 and heads=='BI':
            heads='BE'
        if len(words)==3 and heads=='BII':
            heads='BIE'

        if len(words)==4 and heads=='BIII':
            heads='BIIE'

        if len(words)==5 and heads=='BIIII':
            heads='BIIIE'
        if len(words)==6 and heads=='BIIIII':
            heads='BIIIIE'
        if len(words)==7 and heads=='BIIIIII':
            heads='BIIIIIE'

        if len(words)==8 and heads=='BIIIIIII':
            heads='BIIIIIIE'

        if len(words)==9 and heads=='BIIIIIIII':
            heads='BIIIIIIIE'
        if len(words)==10 and heads=='BIIIIIIIII':
            heads='BBIIIIIIIIE'
        if len(words)==11 and heads=='BIIIIIIIIII':
            heads='BIIIIIIIIIE'
        heads = list(heads)
        revised_tags=[k+'-'+v for k,v in zip(heads,types)]
        revised_tags=['O' if x == 'O-O' else x for x in revised_tags]
        for w,tag in zip(words,revised_tags):

                g.write(w+'\t'+tag+'\n')
        g.write('\n')
g.close()