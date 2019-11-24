import pandas as pd 
from fuzzywuzzy import fuzz

df = pd.read_csv('room_type.csv')

#print(df.head(10))
#print(fuzz.token_set_ratio('Deluxe Room,1 King Bed','Deluxe King Room'))
#print(fuzz.token_set_ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds'))

def get_ratio(row):
    name = row['Expedia']
    name1 = row['Booking.com']
    return fuzz.token_set_ratio(name,name1)

print(len(df[df.apply(get_ratio,axis=1)>70])/len(df))
