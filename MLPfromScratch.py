# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 00:21:52 2020

@author: TOM MASINI
"""


# Projet Long : Implémentation d'un MLP à partir de zéro.


#import pandas as pd
from random import seed
from random import random
from random import randrange
import math as m
from csv import reader




## 1 - Initialisation du réseau neuronal :


# Fonction permettant d'initialiser un network à 3 couches :
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

'''
# Exemple d'initialisation d'un réseau neuronal à 2 neurones en entrée, 1 neurone caché, et 2 neurones en sortie :
seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
	print(layer)
'''


## 2 - Propagation avant :
    
 
# Fonction permettant de calculer l'activation d'un neurone (décrits par weights) en fonction de l'input (inputs) :
def activate(weights, inputs):
	activation = weights[-1] # Le dernier poids (weights[-1]) est le biais du neurone.
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation


# Fonction d'activation sigmoid des sorties de neurones :
def transfer(activation):
	return 1.0 / (1.0 + m.exp(-activation))



# Fonction calculant l'output de la "Rétropopagation Avant d'un network" pour un individu (: row) :
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs



'''
# Test de la fonction de propagation avant :
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}], # Couche cachée à 1 neurone
		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]] # Couche de sortie à 2 neurones
row = [1, 0, None]
output = forward_propagate(network, row) # Calcul la sortie de la ligne 'row' par le réseau 'network'.
print(output)
'''


## 3 - Rétro-Propagation de l'erreur :

# Soit f(x) = 1/(1-exp(-x)) = exp(x)/(1+exp(x)) la fonction sigmoide,
#   sa dérivée est : f'(x) = exp(x)/[1+exp(x)]² = f(x).(1-f(x)). 
#      D'où :
# Fonction calculant la dérivée de la sortie d'un neurone :
def transfer_derivative(output):
	return output * (1.0 - output)



# RétroPropagation de l'erreur et son stockage dans les neurones :
def backward_propagate_error(network, expected): # Pour mieux comprendre, essayer de faire tourner l'algo avec un dessin sur un petit réseau de neurones !
	for i in reversed(range(len(network))): # Pour tout i parcourant les indices décroissants des couches du réseau (i = L, ... , 1).
		layer = network[i] # 'layer' est la couche courante (Attention : l'indexage va de 0 à [len(liste)-1] ).
		errors = list()
		if i != len(network)-1: # Si la couche courante n'est pas la couche de sortie :
			for j in range(len(layer)): # Pour tous les neurones de la couche courante :
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
                    # neuron['delta'] est le signal d'erreur du neurone courant de la couche de sortie i+1 ;
                    # neuron['weights'][j] est le poids qui relie le neurone actuel (de la couche i+1) au neurone j de la couche courante i.
				errors.append(error)
		else: # Si la couche courante est la couche de sortie :
			for j in range(len(layer)): # Pour tous les neurones de la couche de sortie :
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)): # Pour tous les neurones de la couche courante :
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
'''
# Test de la rétroPropagation de l'erreur pour le réseau (2,1,2) initialisé précédemment :
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
	print(layer)
'''



## 4 - Entraînement du réseau :


# Fonction mettant les poids à jour grâce à l'erreur rétroPropagée :
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1] # On enlève le label (qui est aussi stocké dans row).
		if i != 0: # Si la couche courante n'est pas la couche d'entrée :
			inputs = [neuron['output'] for neuron in network[i - 1]] # On redéfini les inputs (qui sont les sorties de la couche précédente).
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] # Màj du poids j du neurone courant.
			neuron['weights'][-1] += l_rate * neuron['delta'] # Màj du biais du neurone courant.
            
            

# Fonction entrainant un réseau durant un nombre fixé d'Epochs :
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train: # Pour chaque individus 'row' dans le trainSet :
			outputs = forward_propagate(network, row) # Calcul l'output de l'individu généré par le network (c'est un vecteur de taille n_outputs).
			expected = [0 for i in range(n_outputs)] # Initialise à 0 le vecteur qui contiendra 1 à l'élément d'indice 'row[-1]' (qui est la classe du label de l'individu row).
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))]) # Calcul l'erreur quadratique faite entre les deux vecteur outputs et expected. 
			backward_propagate_error(network, expected) # Propage l'erreur dans le réseau, et la stocke dans les neurones.
			update_weights(network, row, l_rate) # Màj les poids du réseau (ici, la màj se fait à chaque individu (= online learning)).
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch+1, l_rate, sum_error)) # à chaque époque, affiche l'erreur totale faite sur l'époque courante.




'''
# Test training backprop algorithm :
        
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset])) # set() enlève les doublons, ça permets donc d'avoir le nombre total de classe du label.
network = initialize_network(n_inputs, 2, n_outputs) # Initialise un réseau avec une couche cachée à 2 neurones.
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
	print(layer)
'''




## 5 -  Fonction faisant une prédiction avec un réseau :
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs)) # retourne l'indice de l'élément le plus grand du vecteur 'outputs'.


'''
# Test la prédiction du réseau entrainé :

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]

for row in dataset:
	prediction = predict(network, row)
	print('Vraie valeur =%d, Valeur prédite =%d' % (row[-1], prediction))
'''


# Backpropagation Algorithm With Stochastic Gradient Descent (combine les fonctions précédentes) :
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)





###################### Cette partie n'entre pas dans le cadre d'une implémentation à partir de zéro d'un MLP...
## Les différentes fonctions pour importer le dataSet, normaliser ses valeurs dans [0;1], et tester notre réseau neuronal par validation croisée :

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats


def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])



def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0



def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

###############################################################################
    


# Test de tout le programme sur le dataSET seeds.csv :
seed(1)
# load and prepare data
dataset = load_csv('seeds.csv')
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 50
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


