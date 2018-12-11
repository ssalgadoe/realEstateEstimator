import csv
import numpy as np
import copy
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.svm import SVR
import pickle
from sklearn.externals import joblib

import csv
import googlemaps
gmaps = googlemaps.Client(key='AIzaSyCDjLrnFnVPqfcUb7o3hkDPJjA8LTMaLZo')

def fetchAddress(address):
    try:
        geocode_result = gmaps.geocode(address)
        lat = geocode_result[0]["geometry"]["location"]["lat"]
        lon = geocode_result[0]["geometry"]["location"]["lng"]
    except Exception as er:
        lat = 0.0
        lon = 0.0
        print("lat", lat, "lon", lon, er)
        return lat, lon

    print("lat", lat, "lon", lon)
    return lat, lon


def fetch_neighbours(data, header, lat, lon, postalCode):
    neighbourList = []
    col_post = header.index("postalCode")
    for i,ele in enumerate(data):
        if ele[col_post] == postalCode:
            neighbourList.append(ele)
    return neighbourList

def load_csv_dataset(filename):
    data = []
    with open(filename, 'r') as data_file:
        datareader = csv.reader(data_file, delimiter=',')
        header = next(datareader)
        for line in datareader:
            data.append(line)
    
    
    return header, data

def category_labels(data,header,field):
    unique_set = {}
    col_id = header.index(field)
    temp_set = [ele[col_id] for ele in data]
    unique = set(temp_set)
    for val, key in enumerate(unique):
        unique_set[key] = val
    return unique_set



def check_data_type(data,header,field):
    col_id = header.index(field)
    return all(isinstance(n[col_id], (int, float, complex)) for n in data)    



def data_indexing(data,header,field,dictionary):
    col_id = header.index(field)
    temp = data.copy()
    for i in range(len(temp)):
        try:
            val = dictionary[temp[i][col_id]]
        except:
            val = -100            
        temp[i][col_id] = val
    return temp        


def data_string_remove(data,header,field,phrase, direction):
    col_id = header.index(field)
    temp = data.copy()
    for i in range(len(temp)):
        val = temp[i][col_id]
        pos = val.find(phrase)
        if pos != -1:
            if direction == 'forward':
                val = val[pos+1:].strip()
            elif direction == 'backward':
                val = val[:pos].strip()
        temp[i][col_id] = val
    return temp        
    

def header_list(data):
    header = []
    for ele in data[0]:
        header.append(ele)
    return header

#### To check data is numerical #######
def data_inspection(data,header,field):
    col_id = header.index(field)
    garbage_list = []
    for i in range(len(data)):
        if not data[i][col_id].isdigit():
#            print(data[i])
            garbage_list.append(data[i][col_id])
    return len(garbage_list), set(garbage_list)

def reformat_data(data, header):
    u_address = category_labels(data, header, 'streetAddress')
    data = data_indexing(data, header,'streetAddress', u_address)

    dict_beds = {'Studio':0, '1bd':1,'2bd':2,'3bd':3,'4bd':4,'5bd':5,'6bd':6,'7bd':7,'8bd':8,'9bd':9,'10bd':10,'11bd':11,'12bd':12,'13bd':13, '14bd':14, '15bd':15}
    dict_baths = {'0ba':0, '1ba':1,'2ba':2,'3ba':3,'4ba':4,'5ba':5,'6ba':6,'7ba':7,'8ba':8,'9ba':9,'10ba':10,'11ba':11,'12ba':12,'13ba':13, '14ba':14, '15ba':15}
    data = data_indexing(data, header,'baths', dict_baths)
    data = data_indexing(data, header,'beds', dict_beds)

    data = data_string_remove(data, header,'price', '$','forward')
    data = data_string_remove(data, header,'sqft', 'sqft','backward')
    data = data_string_remove(data, header,'price', '+','backward')

    corrupted_size, corrupted_data = data_inspection(data,header,'price')

    df = pd.DataFrame(data, columns = header)

    numerical_header = ['streetAddress','postalCode','latitute','longitude','beds','baths','sqft', 'price']
    numerical_header = ['postalCode','beds','baths','sqft', 'latitute','longitude','price']
    df = df[numerical_header]

    #to check which column has non-numerical values
    #cols = df.columns[df.dtypes.eq('object')]
    #print(cols)
    df[numerical_header] = df[numerical_header].apply(pd.to_numeric, errors='coerce')

    df.dropna(inplace=True)
    
    #remove the extremes....
    df = df[df['price'] > 100000]
    df = df[df['price'] < 1000000]


    #df = df[:1000]

    X = np.array(df.drop(['price'], axis=1),dtype='float')
    y = np.array(df['price'], dtype='float')
    
    return X, y


def sample_json2csv(sample,header):
    
    samples_csv = []
    for s in sample:
        temp_csv = []
        for ele in header:
            temp_csv.append(s[ele])
        samples_csv.append(temp_csv)
    return samples_csv
    

def prediction_model(X_org,y):
    

   # X = sklearn.preprocessing.scale(X_org)
    #X = sklearn.preprocessing.normalize(X)
    #X = sklearn.preprocessing.minmax_scale(X)
    
    
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X_org)
    


    x_train,x_test, y_train, y_test = train_test_split(X,y, test_size = 0.1)

    model = LinearRegression()
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model = SVR(kernel='linear', C=1e3)
#model = SVR(kernel='poly', C=1e3, degree=2)
#    model = LinearRegression()
    
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    

    return model, scaler




header, data = load_csv_dataset('../houses_filtered.csv')
X_org, y = reformat_data(data,header)
model, scaler =  prediction_model(X_org,y)


model_file = '../my_dope_model.pkl'
scale_file = '../my_dope_scale.pkl'


joblib.dump(model, model_file) 
loaded_model = joblib.load(model_file) 

joblib.dump(scaler, scale_file) 
loaded_scale = joblib.load(scale_file) 

x_test = X_org[:5]
x_test = loaded_scale.fit_transform(x_test)
y_test = y[:5]

raw_sample= [{"streetAddress": "8323 Wilcrest Dr #10020", "addressLoc": "Houston", "addressReg": "TX", "postalCode": "77072", "latitute": "29.686773", "longitude": "-95.56965", "price": "$189900", "beds": "3bd", "baths": "3ba", "sqft": "1503 sqft"},
              {"latitute": "29.686773", "longitude": "-95.56965", "price": "$189900", "beds": "3bd","streetAddress": "8323 Wilcrest Dr #10020", "addressLoc": "Houston", "addressReg": "TX", "postalCode": "77072",  "baths": "3ba", "sqft": "1503 sqft"}]

raw_sample = [{"streetAddress": "5417 Caplin St", "addressLoc": "Houston", "addressReg": "TX", "postalCode": "77026", "latitute": "29.815035", "longitude": "-95.31594", "price": "$87000", "beds": "4bd", "baths": "4ba", "sqft": "2606 sqft"},
{"streetAddress": "6315 E Mystic Mdw", "addressLoc": "Houston", "addressReg": "TX", "postalCode": "77021", "latitute": "29.708738", "longitude": "-95.37922", "price": "$660000", "beds": "3bd", "baths": "1ba", "sqft": "1976 sqft"},
{"streetAddress": "12303 Beauregard Dr", "addressLoc": "Houston", "addressReg": "TX", "postalCode": "77024", "latitute": "29.765789", "longitude": "-95.54463", "price": "$912500", "beds": "5bd", "baths": "1ba", "sqft": "1116 sqft"},
{"streetAddress": "2113 Stuart St", "addressLoc": "Houston", "addressReg": "TX", "postalCode": "77004", "latitute": "29.735182", "longitude": "-95.36756", "price": "$294900", "beds": "2bd", "baths": "2ba", "sqft": "834 sqft"},
{"streetAddress": "3621 Dawson Ln", "addressLoc": "Houston", "addressReg": "TX", "postalCode": "77051", "latitute": "29.654135", "longitude": "-95.36975", "price": "$99000", "beds": "2bd", "baths": "4ba", "sqft": "1150 sqft"},
{"streetAddress": "7110 Brendam Ln", "addressLoc": "Houston", "addressReg": "TX", "postalCode": "77072", "latitute": "29.698895", "longitude": "-95.60962", "price": "$91000", "beds": "5bd", "baths": "6ba", "sqft": "3719 sqft"}]
samples  = sample_json2csv(raw_sample,header)


x_test, y_test = reformat_data(samples,header)
x_test = loaded_scale.fit_transform(x_test)

y_pred = model.predict(x_test)
print(y_pred)
for i in range(len(y_pred)):
    print("Original {0: 10.2f}, Predicted {1: 10.2f}".format(float(y_test[i]), float(y_pred[i])))

address = "6315 E Mystic Mdw, Houston, TX, 77021"
#lat, lon = fetchAddress(address)
ns = fetch_neighbours(data, header,0, 0, "77021")