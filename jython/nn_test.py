"""
nn_test.py - a Jython script by Sam Yeager

Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on abalone_test.py, which is based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

CSV_FILE = 'balance-scale.csv'
CSV_HEADER = False
# The .csv file must go in the ../src/opt/test/ folder
# All data must be numeric.
# All data must
# The response variable (aka. dependent variable, variate, "y value", etc.) must be
#  in the final column.
# If the .csv file has a header, change CSV_HEADER to True

## Default values
# These should be tailored for your data.
OUTPUT_LAYERS = 1    # DO NOT CHANGE OUTPUT_LAYERS!!!
HIDDEN_LAYERS = [5]  # [nodes in hidden layer 1, nodes in hidden layer 2, ...]
NUM_TRIALS = 5       # number of tests to run
NUM_ITERATIONS = 100 # number of data points to train model
SA_TEMP = 10000
SA_COOL = 0.65
GA_POP  = 160
GA_MATE = 20
GA_MUT  = 20

## Test suites
# These should be tailored for testing.
ITERATION_SUITE = [2] + list(map(lambda x: 10**x, range(1, 4, 1)))
TEMP_SUITE = list(map(lambda x: 10**x, range(-11, 10, 1)))
COOL_SUITE = list(map(lambda x: x/100.0, range(0, 105, 5)))
GA_POP_STE = range(20,220,20)
GA_MATE_STE = range(20,GA_POP,20)
GA_MUT_STE = range(20,GA_POP,10)

# CAN ONLY HANDLE 1 OUTPUT!!!
try:
    assert OUTPUT_LAYERS == 1    
except:
    print("Can only handle 1 output layer. Sorry.")
    raise


## Functions

def initialize_instances(file_in):
    """Read the CSV data into a list of instances."""
    instances = []
    
    with open(file_in, "rU") as f_in:
        reader = csv.reader(f_in)

        n_in = None
        if CSV_HEADER:
            _ = reader.next()
    
        for row in reader:
            if not n_in:
                n_in = len(row[:-1])
            instance = Instance([float(value) for value in row[:-1]])
            try:
                instance.setLabel(Instance(float(row[-1])))
            except:
                print("Response variable must be numeric.")
                return
            instances.append(instance)

    return instances, n_in


def train(oa, network, oaName, instances, measure, n_iterations):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
#     print "\nError results for %s\n---------------------------" % (oaName,)

    for _ in xrange(n_iterations):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

#         print "%0.03f" % error


def nn_test(csv_in, oa_name=None, 
            output_layers = OUTPUT_LAYERS, hidden_layers = HIDDEN_LAYERS,
            tst_lbl = "N/A", lvl_num = "N/A",
            n_iter = NUM_ITERATIONS, 
            sa_temp = SA_TEMP, sa_cool = SA_COOL, 
            ga_pop = GA_POP, ga_mate = GA_MATE, ga_mut = GA_MUT):
    """Run algorithms on the dataset in the csv_in file.
    
    Inputs:
      csv_in:  str   - the filename of the .csv with the data
      oa_name: str   - a specific optimization algorithm name. must be 'RHC', 'SA', or 'GA'
      output_layers: - this should be always 1
      hidden_layers: [num_nodes_in_layer_1, num_nodes_in_layer_2, ...] 
        - list of the number of nodes in each hidden layer
      tst_lbl: str   - the label for this test
      n_iter:  int   - number of data points to train with
      sa_temp: int or float - the 'temperature' at which simulated annealing occurs
      sa_cool: float - the amount of 'cooling' in simulated annealing; 0 <= sa_cool <= 1
      ga_pop:  int   - number of 'parents' to keep in each generation, in genetic algorithm
      ga_mate: int   - amount of crossover that occurs between parents, in genetic algorithm
      ga_mut:  int   - amount of mutation that occurs among parents, in genetic algorithm
      
    Outputs:
      output_{time}.csv - a .csv file with all the test data
    """
    
    assert n_iter >= 1
    
    file_in = os.path.join("..", "src", "opt", "test", csv_in)
    

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    
    oa = []  # OptimizationAlgorithm
    oa_names = ['RHC', 'SA', 'GA']
    if oa_name:
        assert oa_name in oa_names
        oa_names = [oa_name]
    
    if 'SA' in oa_names:
        assert sa_cool >= 0 and sa_cool <= 1 and sa_temp > 0
    else:
        sa_temp = sa_cool = None
    if 'GA' in oa_names:
        assert ga_pop > 0 and ga_mate**2 * ga_mut > 0 # ensures everything is strictly positive
        if ga_mate > ga_pop or ga_mut > ga_pop:
            return
    else:
        ga_pop = ga_mate = ga_mut = None
        
    params = {'RHC': {'algo': RandomizedHillClimbing,
                      'args': []},
              'SA': {'algo': SimulatedAnnealing,
                     'args': [sa_temp, sa_cool]},
              'GA': {'algo': StandardGeneticAlgorithm,
                     'args': [ga_pop, ga_mate, ga_mut]}}

    instances, input_layers = initialize_instances(file_in)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(instances)

    for trial_num in range(1, NUM_TRIALS+1):
        results = ""
    
        for i, oa_name in enumerate(oa_names):
            classification_network = factory.createClassificationNetwork([input_layers]+hidden_layers+[output_layers])
            networks.append(classification_network)
            nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))
    
            # Example idea of line below
            #   oa.append(StandardGeneticAlgorithm(ga_pop, ga_mate, ga_mut, nnop[2]))
            oa.append(params[oa_name]['algo'](*(params[oa_name]['args']+[nnop[i]])))
    
    
        for i, oa_name in enumerate(oa_names):
            start = time.time()
            correct = 0
            incorrect = 0
    
            train(oa[i], networks[i], oa_names[i], instances, measure, n_iter)
            end = time.time()
            training_time = end - start
    
            optimal_instance = oa[i].getOptimal()
            networks[i].setWeights(optimal_instance.getData())
    
            start = time.time()
            for instance in instances:
                networks[i].setInputValues(instance.getData())
                networks[i].run()
                
    
                predicted = instance.getLabel().getContinuous()
                actual = networks[i].getOutputValues().get(0)
    
                if abs(predicted - actual) < 0.5:
                    correct += 1
                else:
                    incorrect += 1
    
            end = time.time()
            testing_time = end - start
            
            class_rate=float(correct)/(correct+incorrect)*100.0
    
            results += "\nResults for %s: \nCorrectly classified %d instances." % (oa_name, correct)
            results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, class_rate)
            results += "\nTraining time: %0.03f seconds" % (training_time,)
            results += "\nTesting time: %0.03f seconds\n" % (testing_time,)
        
            # This file outputs to ../jython/ not to ../src/
            with open("output_{}.csv".format(now), 'a') as f_out:
                csvwriter = csv.writer(f_out)
                row_meta = [tst_lbl, lvl_num, trial_num] 
                row_input = [oa_name, n_iter, sa_temp, sa_cool, ga_pop, ga_mate, ga_mut]
                row_output = [class_rate, training_time]
                csvwriter.writerow(row_meta+row_input+row_output)
                f_out.flush()
            
        print results


if __name__ == "__main__":
    csv_file = CSV_FILE
    now = time.strftime('%m%d-%H%M', time.localtime())
    with open("output_{}.csv".format(now), 'a') as f_out: # This file outputs to ../jython/ not to ../src/
        csvwriter = csv.writer(f_out)
        csvwriter.writerow(['test_name', 'level_num', 'trial_number',
                            'oa_mame', 'num_iters', 
                            'sa_temp', 'sa_cool', 
                            'ga_pop', 'ga_mate', 'ga_mut',
                            'classification_pct', 'training_time'])

    for level, num_iter in enumerate(ITERATION_SUITE):
        nn_test(csv_file, n_iter=num_iter, tst_lbl='iteration_test', lvl_num = level)
    for level, temp in enumerate(TEMP_SUITE):
        nn_test(csv_file, sa_temp=temp, oa_name="SA", tst_lbl='sa_temp_test', lvl_num = level)
    for level, cool in enumerate(COOL_SUITE):
        nn_test(csv_file, sa_cool=cool, oa_name="SA", tst_lbl='sa_cool_test', lvl_num = level)
    for level, parents in enumerate(GA_POP_STE):
        nn_test(csv_file, ga_pop=parents, oa_name="GA", tst_lbl='ga_pop_test', lvl_num = level)
    for level, mate_rate in enumerate(GA_MATE_STE):
        nn_test(csv_file, ga_mate=mate_rate, oa_name="GA", tst_lbl='ga_mate_test', lvl_num = level)
    for level, mutations in enumerate(GA_MATE_STE):
        nn_test(csv_file, ga_mut=mutations, oa_name="GA", tst_lbl='ga_mut_test', lvl_num = level)
