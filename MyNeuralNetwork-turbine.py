from sklearn.model_selection import train_test_split
import pandas as pd

from MyNeuralNetwork import MyNeuralNetwork

fileName='A1-synthetic/A1-synthetic-normalized.csv'

file_content= pd.read_csv(fileName,delimiter=',')
label="synthetic"

test_percentage=0.20
validation_percentage=0.20

activation_functions=['sigmoid','relu','linear','tanh']
momentums=[0.01,0.1,0.3]
learning_rates=[0.3,0.5,0.9]
eps=[50,100]

layers = [4,3,1] 

input_data = file_content.iloc[:, :-1]
target_data = file_content.iloc[:, -1]

input_data=input_data.values.tolist()
target_data=target_data.values.tolist()

# Split the data into training and test sets
validation_input_data, input_test, validation_target_data, target_test = train_test_split(input_data, target_data, test_size=test_percentage, random_state=42)

best_mape=0
best_epoch=0
best_momentum=0
best_lr=0
best_activation=''

for epo in eps:
    for activation_function in activation_functions:
        for momentum in momentums:
            for learning_rate in learning_rates:
                
                # layers include input layer + hidden layers + output layer
                nn = MyNeuralNetwork(layers, epochs=epo, learning_rate=learning_rate, momentum=momentum,activation_function=activation_function,validation_percentage=validation_percentage)

                nn.fit(validation_input_data, validation_target_data,False)

                # Predict using the test set
                test_prediction = nn.predict(input_test)
                mape = nn.calculate_mape(target_test, test_prediction)

                if best_mape==0 or mape<best_mape:
                    best_mape=mape
                    best_epoch=epo
                    best_momentum=momentum
                    best_lr=learning_rate
                    best_activation=activation_function

                print(label,f'MAPE: {mape:.2f}%','epoch:',epo,'activation:',activation_function,'lr:', learning_rate,'momentum:',momentum)

print(label,"Best parameters: mape",best_mape,'epoch:',best_epoch,'activation:',best_activation,'lr:', best_lr,'momentum:',best_momentum)





