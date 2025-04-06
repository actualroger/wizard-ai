
import matplotlib.pyplot as plt
import numpy as np

# plot function
def plotCurves(arr_list, legend_list, color_list, x_values = [], upper_bound = [], upper_bound_label: str = 'Upper Bound', xlabel: str = '', ylabel: str = ''):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        upper_bound (numpy array): array contains the best possible rewards for 2000 runs. the shape should be (2000,)
        ylabel (string): label of the Y axis
        
        Note that, make sure the elements in the arr_list, legend_list and color_list are associated with each other correctly. 
        Do not forget to change the ylabel for different plots.
        
        To plot the upper bound for % Optimal action figure, set upper_bound = np.ones(num_step), where num_step is the number of steps.
    """

    # cast input if necessary
    if isinstance(arr_list, list):
        arr_list = np.array(arr_list)
    # if isinstance(upper_bound, list):
    #     upper_bound = np.array(upper_bound)

    # set the figure type
    # plt.clf()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # set labels
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)

    # plot results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        x_data = range(arr.shape[1]) if len(x_values) == 0 else x_values
        # compute the standard error
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(x_data, arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err = 1.96 * arr_err
        ax.fill_between(x_data, arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3, color=color)
        # save the plot handle
        h_list.append(h) 
    
    # plot the upper bound
    if len(upper_bound) > 0:
        x_data = range(len(upper_bound)) if len(x_values) == 0 else x_values
        h, = ax.plot(x_data, upper_bound, color='k', linestyle='--', label=upper_bound_label)
        h_list.append(h)
    
    # plot legends
    ax.legend(handles=h_list)  
    plt.show()

# plot, but autogenerate agent names and colors
def plotCurvesAutolabel(arr_list, **kwargs):
    numPlayers = len(arr_list)
    plotColors = ['r','g','b','k','y','c']
    plotCurves(arr_list, ['Agent %d'%i for i in range(numPlayers)], plotColors[:numPlayers], **kwargs)
    # TODO this should be able to infer numPlayers

# plot scores vs. round for each agent
def plotAgentScores(scores):
    numPlayers = len(scores)
    roundNumbers = range(1, 60 // numPlayers + 1)
    expectedScores = [20 + 10 * (round / (0.0 + numPlayers)) for round in roundNumbers]
    plotCurvesAutolabel(scores, x_values=roundNumbers, xlabel='Round', ylabel='Score', upper_bound=expectedScores, upper_bound_label='Nominal')

# trim all dimensions of an n-dimensional list
def trimDims(L):
    if isinstance(L, list):
        if isinstance(L[0], list):
            minLen = min([len(l) for l in L])
            L = [l[:minLen] for l in L]
        L = [trimDims(l) for l in L]
    return L
