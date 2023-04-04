import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--source_folder', type=str,default='./visualization/example')
parser.add_argument('--target_folder', type=str,default='./results/example')
args = parser.parse_args()
os.system("mkdir %s" %args.target_folder)

# outdirs =os.listdir(args.source_folder)
# for a in outdirs:
dirs = os.listdir(args.source_folder)
for each in dirs:
    name = each.split('.')
    temp_path = os.path.join(args.source_folder,each)
    out_path = os.path.join(args.target_folder,name[0]+'.png')
    os.system("weblogo -F png -X False -Y False -c classic -U probability --fineprint '' --resolution 300 <%s> %s" %(temp_path,out_path))
