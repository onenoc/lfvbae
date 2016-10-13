h = ['hello', 'there']
f=open('myfile','w')
for item in h:
    f.write(item+'\n')
f.close()

