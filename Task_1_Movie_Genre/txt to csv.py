import pandas as pd

# Replace 'path_to_train.txt' and 'path_to_test.txt' with the actual paths to your train and test text files
train_file_path = 'Train.txt'
test_file_path = 'Test.txt'
_
# Function to read data in chunks
def read_data_in_chunks(file_path):
    data = {'title': [], 'genre': [], 'plot': []}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line using ':::'
            parts = line.strip().split(':::')
            
            # Process each line with three parts
            if len(parts) == 4:
                data['title'].append(parts[1].strip())
                data['genre'].append(parts[2].strip())
                data['plot'].append(parts[3].strip())

    # Create a DataFrame from the data
    return pd.DataFrame(data)

# Read the training data in chunks
train_data = read_data_in_chunks(train_file_path)

# Save the training data to a CSV file
train_data.to_csv('train_data.csv', index=False)

# Function to read testing data
def read_test_data(file_path):
    data = {'title': [], 'plot': []}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split each line using ':::'
            parts = line.strip().split(':::')
            
            # Process each line with two parts (title and plot)
            if len(parts) == 3:
                data['title'].append(parts[1].strip())
                data['plot'].append(parts[2].strip())

    # Create a DataFrame from the data
    return pd.DataFrame(data)

# Read the testing data
test_data = read_test_data(test_file_path)

# Save the testing data to a CSV file
test_data.to_csv('test_data.csv', index=False)
