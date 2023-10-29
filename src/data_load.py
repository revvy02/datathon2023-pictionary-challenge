import pandas as pd
import os
import json 
from itertools import islice


def loadData():
    # load data
    columns = ['Image', 'Label']
    data = pd.DataFrame(columns=columns)

    # Cycle through all the image categories and pick the first 500 images
    num_images = 20
    current_directory = os.getcwd()

    dataset_path = os.path.join(current_directory, '..', 'simplified')

    file_names = os.listdir(dataset_path)

    count = 0
    for file_name in file_names:
        with open(dataset_path + "/" + file_name, 'r') as file:
            first_500_elements = list(islice(map(json.loads, file), num_images))

            for element in first_500_elements:
                word = element['word']
                drawing = element['drawing']

                # Add a new row to the existing DataFrame
                new_row = {'Image': drawing, 'Label': word}
                data.loc[len(data.index)] = new_row 
        count+= 1
        if count == 10:
            break
    
    file = current_directory + "/data2.csv"
    data.to_csv(file)

loadData()