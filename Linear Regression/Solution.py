import numpy as np

# This function computes the weights for a given predictor and response data set
def compute_weights(predictor, response):
    # Consider only the first half of the predictor data
    predictor = predictor[:len(predictor)//2]

    # Convert the predictors to numpy array of float data, and add a columns of 1's in the first position
    predictor = np.insert(predictor, 0, [1] * len(predictor), axis = 1).astype(np.float)

    # Creatte transpose of the predictors
    predictor_t = predictor.transpose()

    # Conver the first half of the responses to numpy array of float data
    response = np.array(response[:len(response)//2], dtype= 'O').astype(np.float)

    # Compute the weight of the provided predictors and responses, according to the equation W = (Xt * X)-1 Xt * Y
    return np.matmul(np.linalg.inv(np.matmul(predictor_t, predictor)), np.matmul(predictor_t, response))

# This function computes the error for a given predictor, response and their weights set
def compute_error(predictor, response, weights):
    # Conver the second half of the predictors and provided predictions to a numpy array of float datatype
    predictor = np.array(predictor[len(predictor)//2+1:], dtype= 'O').astype(np.float)
    provided_predictions = np.array(response[len(response)//2+1:], dtype= 'O').astype(np.float)

    # Compute the weights using the fomula W0 + Sum of WnXn
    computed_predictions = weights[0] + np.matmul(predictor, weights[1:])

    # Calculate the responses using the computed weights
    return np.square(np.subtract(computed_predictions, provided_predictions)).sum()

if __name__ == "__main__":
    # Read the data from the predictor and response files
    with open('pred1.dat') as file:
        pred1 = [ [i.strip() for i in row.split(' ')] for row in file]
    with open('pred2.dat') as file:
        pred2 = [ [i.strip() for i in row.split(' ')] for row in file]
    with open('resp1.dat') as file:
        resp1 = [ [i.strip() for i in row.split(' ')] for row in file]
    with open('resp2.dat') as file:
        resp2 = [ [i.strip() for i in row.split(' ')] for row in file]

    # Part 1 - Compute the weights of the given set of data
    # Calculate weights for the first set of data
    computedWeights1 = compute_weights(pred1, resp1) 

    # Calculate weights for the second set of data
    computedWeights2 = compute_weights(pred2, resp2)

    # Part 2 - Compute the errors of the given set of data and their computed weights in part 1
    # Calculate the error for the first set of data
    error1 = compute_error(pred1, resp1, computedWeights1)

    # Calculate the error for the second set of data
    error2 = compute_error(pred2, resp2, computedWeights2)

    # Print the calcuated erorrs of each dataset
    print(error1, error2) #error1 = 5.713466807813311, error2 = 10140348.245954204