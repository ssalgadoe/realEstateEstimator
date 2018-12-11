import json
import numpy as np


def load_jl_records(filename):
    houses = []
    with open(filename) as data_file:
        for i,line in enumerate(data_file):
            if len(line.strip()) > 0:
                houses.append(json.loads(line.strip()))
        
    print("Processed total {0}".format(i))
    return houses

def save_jl_records(filename, data):
    header = header_list(data)
    header_str = ",".join(header) + "\n"
    with open(filename,'w') as data_file:
        data_file.write(header_str)
        for idx, line in enumerate(data):
            line_array = []
            for i in range(len(header)):
                line_array.append(line[header[i]])
            line_str = ",".join([str(e).replace(",","") for e in line_array]) + "\n"
            data_file.write(line_str)
        

def json2csv_conversion(input_file, output_file, duplicate_field):
    raw_data = load_jl_records(input_file)
    filtered_data = remove_duplicates(raw_data, duplicate_field)
    print("Total valid records {0}".format(len(filtered_data)))
    save_jl_records(output_file, filtered_data)
    print("data converstion is done!!!")



def remove_duplicates(data, field):
    filtered = []
    temp = []
    duplicates = 0
    for i in range(len(data)):
        value = data[i][field]
        if value not in temp:
            temp.append(value)
            filtered.append(data[i])
        else:
            duplicates+=1
    
    print("Nr of Duplicate records {0}".format(duplicates))                
    
    return filtered


def header_list(data):
    header = []
    for ele in data[0]:
        header.append(str(ele).replace(",",""))
    return header


json2csv_conversion('../houses.jl','../houses_filtered.csv','streetAddress')
#houses = load_jl_records('../houses_sample.jl')
#header = header_list(houses)
#print(header)
#filtered = remove_duplicates(houses, 'streetAddress')
#print("data len {0}, filtered len {1}".format(len(houses), len(filtered)))
#
#u_beds = create_labels(filtered, 'beds')
#d_type = check_data_type(filtered, 'beds')
#print(d_type)
#indexed = data_indexing(filtered, 'beds', u_beds)
#d_type = check_data_type(indexed, 'beds')
#print(d_type)
#save_jl_records("test", filtered)