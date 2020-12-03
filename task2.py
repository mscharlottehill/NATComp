import numpy as np
import numpy.random as rnd
import sklearn.neural_network as nn # not ideal but I see no reason why an MLPClassifier can't be used for Tasks 2 and 3
# GA code is borrowed from a cited source (will add to report)
# kicks up an error in SKLearn, not sure why, will debug

def sort_data():
    # load the data, shuffle it and get a 50:50 TTS
    data = np.loadtxt("two_spirals.dat")
    halfway = int(0.5*len(data))
    np.random.shuffle(data)
    training = data[:halfway]
    testing = data[halfway:]
    return training, testing

def ready_for_input(dataset):
    labels = dataset[:,2]
    inputs = dataset[:,:2]
    xsq = np.square(inputs[:,0])
    ysq = np.square(inputs[:,1])
    sinx = np.sin(inputs[:,0])
    siny = np.sin(inputs[:,1])
    input_features = np.transpose(np.vstack((inputs[:,0],inputs[:,1],xsq,ysq,sinx,siny)))
    return input_features, labels


node_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8] # reasonable range of layer sizes

def initialize():
   population = []
   for i in range(10):
       this_individual = [0, 0, 0, 0]
       for j in range (4):
           this_individual[j] = np.random.choice(node_numbers)
       population.append(this_individual)
   return np.array(population) # get 10 (in this case) random initial structures

def getloss(pred, labs):
    err = 0.0
    for i in range(len(pred)):
        curr_err = (pred[i]-labs[i])**2
        err = err + curr_err
    return err

def cost_function(array, data, labels):
    remove_zeros = array[array != 0]
    structure = tuple(remove_zeros)
    neuralnet = nn.MLPClassifier(structure, solver='sgd')
    neuralnet = neuralnet.fit(data, labels)
    predictions = neuralnet.predict(data)
    loss = getloss(predictions, labels)
    return loss

def eval_fit_population(population):
   result = {}
   fit_vals_lst = []
   solutions = []
   for solution in population:
      fit_vals_lst.append(cost_function(solution, train_feats, train_labels))
      solutions.append(solution)
   result["fit_vals"] = fit_vals_lst
   min_wgh = [np.max(list(result["fit_vals"]))-i for i in list(result["fit_vals"])]
   result["fit_wgh"]  = [i/sum(min_wgh) for i in min_wgh]
   result["solution"] = np.array(solutions)
   return result

def pickOne(population):
   fit_bag_evals = eval_fit_population(population)
   a=True
   while a:
      rnIndex = rnd.randint(0, len(pop_bag)-1)
      rnPick  = fit_bag_evals["fit_wgh"][rnIndex]
      r = rnd.random()
      if  r <= rnPick:
         pickedSol = fit_bag_evals["solution"][rnIndex]
         a = False
   return pickedSol

def crossover(solA, solB):
   n     = len(solA)
   child = [np.nan for i in range(n)]
   num_els = np.ceil(n*(rnd.randint(10,90)/100))
   str_pnt = rnd.randint(0, n-2)
   end_pnt = n if int(str_pnt+num_els) > n else int(str_pnt+num_els)
   blockA = list(solA[str_pnt:end_pnt])
   child[str_pnt:end_pnt] = blockA
   for i in range(n):
      if list(blockA).count(solB[i]) == 0:
         for j in range(n):
            if np.isnan(child[j]):
               child[j] = solB[i]
               break
   return child

def mutation(sol):
   n = len(sol)
   pos_1 = rnd.randint(0,n-1)
   pos_2 = rnd.randint(0,n-1)
   result = swap(sol, pos_1, pos_2)
   return result

def swap(sol, posA, posB):
   result = sol.copy()
   elA = sol[posA]
   elB = sol[posB]
   result[posA] = elB
   result[posB] = elA
   return result

train, test = sort_data()
train_feats, train_labels = ready_for_input(train)
test_feats, test_labels = ready_for_input(test)

# Create the initial population bag
pop_bag  = initialize()
# Iterate over all generations
for g in range(20):
   # Calculate the fitness of elements in population bag
   pop_bag_fit = eval_fit_population(pop_bag)
   # Best individual so far
   best_fit = np.min(pop_bag_fit["fit_vals"])
   best_fit_index = pop_bag_fit["fit_vals"].index(best_fit)
   best_solution  = pop_bag_fit["solution"][best_fit_index]
   # Check if we have a new best
   if g == 0:
      best_fit_global      = best_fit
      best_solution_global = best_solution
   else:
      if best_fit <= best_fit_global:
         best_fit_global      = best_fit
         best_solution_global = best_solution
   # Create the new population bag
   new_pop_bag = []
   for i in range(10):
      # Pick 2 parents from the bag
      pA = pickOne(pop_bag)
      pB = pickOne(pop_bag)
      new_element = pA
      # Crossover the parents
      if rnd.random() <= 0.87:
         new_element = crossover(pA, pB)
      # Mutate the child
      if rnd.random() <= 0.7:
         new_element = mutation(new_element)
      # Append the child to the bag
      new_pop_bag.append(new_element)
   # Set the new bag as the population bag
   pop_bag = np.array(new_pop_bag)
   print("g="+str(g))
#return best_fit_global, best_solution_global
