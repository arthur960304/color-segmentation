import os
import numpy as np

train_folder = "trainset"

# dictionary for the pixel rgb data
data_dict = {
    'COLOR_STOP_SIGN_RED': np.empty(shape=[0, 3]),
    'COLOR_OTHER_RED'    : np.empty(shape=[0, 3]),
    'COLOR_BROWN'        : np.empty(shape=[0, 3]),
    'COLOR_ORANGE'       : np.empty(shape=[0, 3]),
    'COLOR_BLUE'         : np.empty(shape=[0, 3]),
    'COLOR_OTHER'        : np.empty(shape=[0, 3])
}

# dictionary for the pixel mean values, converting to numpy arrays for faster computation.
mean_dict = {
    'COLOR_STOP_SIGN_RED': np.array([0, 0, 0]),
    'COLOR_OTHER_RED'    : np.array([0, 0, 0]),
    'COLOR_BROWN'        : np.array([0, 0, 0]),
    'COLOR_ORANGE'       : np.array([0, 0, 0]),
    'COLOR_BLUE'         : np.array([0, 0, 0]),
    'COLOR_OTHER'        : np.array([0, 0, 0])
}

# dictionary for the pixel covariance values, converting to numpy arrays for faster computation.
cov_dict = {
    'COLOR_STOP_SIGN_RED': np.empty(shape=[3, 3]),
    'COLOR_OTHER_RED'    : np.empty(shape=[3, 3]),
    'COLOR_BROWN'        : np.empty(shape=[3, 3]),
    'COLOR_ORANGE'       : np.empty(shape=[3, 3]),
    'COLOR_BLUE'         : np.empty(shape=[3, 3]),
    'COLOR_OTHER'        : np.empty(shape=[3, 3])
}

# dictionary to store prior probability of each color classes
prior_dict = {
    'COLOR_STOP_SIGN_RED': 0,
    'COLOR_OTHER_RED'    : 0,
    'COLOR_BROWN'        : 0,
    'COLOR_ORANGE'       : 0,
    'COLOR_BLUE'         : 0,
    'COLOR_OTHER'        : 0
}

def get_mean_cov(class_name):
    """Calculate mean and covariance of different color classes
    
    Args:
        class_name(string): COLOR_STOP_SIGN_RED, COLOR_OTHER_RED, COLOR_BROWN, COLOR_ORANGE, COLOR_BLUE, COLOR_OTHER
    
    Returns:
        mean and covariance dictionary
    """
    # load npz files
    for file in os.listdir(train_folder):
        if file.endswith(".npz"):
            file_path = os.path.join(train_folder, file)
            npz_file = np.load(file_path)
            
            # if contains class data
            if(len(npz_file[class_name])):
                data_dict[class_name] = np.append(data_dict[class_name], np.array(npz_file[class_name]), axis=0)
                
    mean_dict[class_name] = np.mean(data_dict[class_name], axis=0)
    cov_dict[class_name]  = np.cov(data_dict[class_name].T)

    return mean_dict, cov_dict
    
def prior(class_name):
    """Calculate prior probablity of different color classes for Naive Bayes Classifier
    
    Args:
        class_name(string): COLOR_STOP_SIGN_RED, COLOR_OTHER_RED, COLOR_BROWN, COLOR_ORANGE, COLOR_BLUE, COLOR_OTHER
    
    Returns:
        prior dictionary
    """
    total_points = len(data_dict['COLOR_STOP_SIGN_RED']) + len(data_dict['COLOR_OTHER_RED']) \
                   + len(data_dict['COLOR_BROWN']) + len(data_dict['COLOR_ORANGE']) \
                   + len(data_dict['COLOR_BLUE']) + len(data_dict['COLOR_OTHER'])
    
    prior_dict[class_name] = len(data_dict[class_name]) / total_points

    return prior_dict



if __name__ == '__main__':
    # calculate gaussian parameters for different classes
    get_mean_cov('COLOR_STOP_SIGN_RED')
    get_mean_cov('COLOR_OTHER_RED')
    get_mean_cov('COLOR_BROWN')
    get_mean_cov('COLOR_ORANGE')
    get_mean_cov('COLOR_BLUE')
    get_mean_cov('COLOR_OTHER')

    # calculate prior probability for different classes
    prior('COLOR_STOP_SIGN_RED')
    prior('COLOR_OTHER_RED')
    prior('COLOR_BROWN')
    prior('COLOR_ORANGE')
    prior('COLOR_BLUE')
    prior('COLOR_OTHER')

    print(mean_dict)
    print(cov_dict)
    print(prior_dict)