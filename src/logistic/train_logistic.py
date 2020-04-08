import os
import numpy as np
import matplotlib.pyplot as plt
from logistic_model import LogisticRegression

train_folder = "trainset"


# dictionary for the pixel rgb data
data_dict = {
    'COLOR_RED'   : np.empty(shape=[0, 3]),
    'COLOR_OTHER' : np.empty(shape=[0, 3])
}

label_dict = {
    'COLOR_RED'   : np.empty(shape=[0, 1]),   # 1
    'COLOR_OTHER' : np.empty(shape=[0, 1])    # 0
}


def read_data(class_name):
    """Read data of different color classes
    
    Args:
        class_name(string): COLOR_STOP_SIGN_RED, COLOR_OTHER_RED, COLOR_BROWN, COLOR_ORANGE, COLOR_BLUE, COLOR_OTHER
    """
    # load npz files
    for file in os.listdir(train_folder):
        if file.endswith(".npz"):
            file_path = os.path.join(train_folder, file)
            npz_file = np.load(file_path)
            data_len = len(npz_file[class_name])
            
            # if contains class data
            if(len(npz_file[class_name])):
                if(class_name=='COLOR_STOP_SIGN_RED' or class_name=='COLOR_OTHER_RED'):
                    data_dict['COLOR_RED'] = np.append(data_dict['COLOR_RED'], np.array(npz_file[class_name]), axis=0)
                    label_dict['COLOR_RED'] = np.append(label_dict['COLOR_RED'], [1]*data_len)
                else:
                    data_dict['COLOR_OTHER'] = np.append(data_dict['COLOR_OTHER'], np.array(npz_file[class_name]), axis=0)
                    label_dict['COLOR_OTHER'] = np.append(label_dict['COLOR_OTHER'], [0]*data_len)
                    

def create_input_label(data_dict, label_dict):
    """Create suitable input and label to the network
    
    Args:
        data_dict(dict): Dictionary contains data of different classes
        label_dict(dict): Dictionary contains label of different classes
        
    Returns:
        x: Input for the model
        y: Label for the model
    """
    x = np.empty(shape=[0, 3])
    y = np.empty(shape=[0, 1])
    
    for _, v in data_dict.items():
        x = np.concatenate((x,v))
        
    for _, v in label_dict.items():
        v = np.reshape(v, (-1,1))
        y = np.concatenate((y,v))
        
    return x, y


if __name__ == '__main__':
    # read class data
    print("Reading data...")
    read_data('COLOR_STOP_SIGN_RED')
    read_data('COLOR_OTHER_RED')
    read_data('COLOR_BROWN')
    read_data('COLOR_ORANGE')
    read_data('COLOR_BLUE')
    read_data('COLOR_OTHER')

    # create suitable input and label format for the model
    print("Creating input and label for the model...")
    x, y = create_input_label(data_dict, label_dict)

    # create and train the model
    print("Training the model...")
    model = LogisticRegression(n_features=3)
    w, b = model.weightInit()
    coeff, gradient, costs = model.train(w, b, x, y, lr=0.001, no_iterations=50)
    print(coeff)

    # plot the training loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Cost reduction over time')
    plt.show()