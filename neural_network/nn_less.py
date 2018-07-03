import numpy as np

def af(x):
    if x >= 0.5:
        return 1
    else:
        return 0

def predict(vodka, rain, friend):
    input = np.array([vodka, rain, friend], ndmin=2).T
    print(input)
    weight_input_to_hidden1 = [0.25, 0.25, 0] # к первому нейрону второго слоя
    weight_input_to_hidden2 = [0.5, -0.4, 0.9] # к второму нейрону второго слоя
    weight_input_to_hidden = np.array([weight_input_to_hidden1, weight_input_to_hidden2])

    '''Этот список представляет собой первый нейрон второго слоя, в который входят сиглалы от нейронов первого слоя от 1 2 3 нейрона'''
    print(weight_input_to_hidden[0], "Этот список представляет собой первый нейрон второго слоя, в который входят сиглалы от нейронов первого слоя от 1 2 3 нейрона")

    '''---------------------------------------------------------------------------------------------------------'''
    '''этот список представляет из себя единственный нейрон выходного слоя, к кторому идут синапсы от первого и второго нейрона второго слоя (скрытого слоя)'''
    weight_hidden_to_output = np.array([-1, 1])

    hidden_input = np.dot(weight_input_to_hidden, input)
    hidden_output = np.array(list(map(af, hidden_input)))

    final_input = np.dot(weight_hidden_to_output, hidden_output)
    print(af(final_input))

predict(1,0,1)