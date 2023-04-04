import os
pos='./positive/'
neg='./negative/'

filenames=os.listdir(pos)
for eachfile in filenames:
	output=[]
	#print(eachfile.split('.'))
	#exit()
	output_file=open(eachfile.split('.')[0]+'.txt','a')

	f=open(os.path.join(pos+eachfile),'r')
	i=0
	for each in f:
		i+=1
		if i%2 == 0:
			output_file.write(str(each[200:406])+'\n')

filenames=os.listdir(neg)
for eachfile in filenames:
	output=[]
	output_file=open(eachfile.split('.')[0]+'.txt','a')

	f=open(os.path.join(neg+eachfile),'r')
	i=0
	for each in f:
		i+=1
		if i%2 == 0:
			output_file.write(str(each[200:406])+'\n')