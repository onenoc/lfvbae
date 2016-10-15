h = ['hello', 'there']
f=open('myfile.txt','w')
for item in h:
    f.write(item+'\n')
f.close()

