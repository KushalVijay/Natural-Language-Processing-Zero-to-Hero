import csv
txt_file = r"SMSSpamCollection.txt"
csv_file = r"SMSSpamCollection.csv"


in_txt = csv.reader(open(txt_file,"r"),delimiter = '\t')
out_csv = csv.writer(open(csv_file,'w'))

out_csv.writerows(in_txt)
