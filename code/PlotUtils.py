
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

# plot function
def plotCurves(arr_list,
               legend_list,
               color_list,
               x_values = [],
               upper_bound = [],
               upper_bound_label: str = 'Upper Bound',
               xlabel: str = '',
               ylabel: str = '',
               show: bool = True,
               filename: str = None):
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

    # write to file if desired
    if filename is not None:
        plt.savefig(filename)

    # show if desired
    if show:
        plt.show()
    else:
        plt.clf()

# plot, but autogenerate agent names and colors
def plotCurvesAutolabel(arr_list, **kwargs):
    numPlayers = len(arr_list)
    plotColors = ['r','g','b','k','y','c']
    plotCurves(arr_list, ['Agent %d'%i for i in range(numPlayers)], plotColors[:numPlayers], **kwargs)

# plot scores vs. round for each agent
def plotAgentScores(scores, **kwargs):
    numPlayers = len(scores)
    roundNumbers = range(1, 60 // numPlayers + 1)
    expectedScores = [20 + 10 * (round / (0.0 + numPlayers)) for round in roundNumbers]
    plotCurvesAutolabel(scores, x_values=roundNumbers, xlabel='Round', ylabel='Score', upper_bound=expectedScores, upper_bound_label='Nominal', **kwargs)

# trim all dimensions of an n-dimensional list
def trimDims(L):
    # returns a list [[a], [b,c], [d,e,f,g]] of the lengths of elements by depth
    def getLengthTree(lst):
        if isinstance(lst, list): # if this is a list
            thisLengths = [[len(lst)]]
            if isinstance(lst[0], list): # recurse if necessary
                for el in lst:
                    subLengths = getLengthTree(el)
                    for j in range(len(subLengths)):
                        if j + 1 >= len(thisLengths):
                            thisLengths.append(subLengths[j]) # extend tree
                        else:
                            thisLengths[j + 1].extend(subLengths[j]) # merge into tree
            return thisLengths
        return 1  # not a list

    treeLengths = getLengthTree(L)
    treeLengths = [min(d) for d in treeLengths] # minimum of each length by depth

    def trimDimension(lst, treeLengths, depth=0):
        if depth >= len(treeLengths):
            return lst
        return [trimDimension(el, treeLengths, depth+1) for el in lst[:treeLengths[depth]]]

    return trimDimension(L, treeLengths)

# plot training losses
def plotTrainingLosses(train_losses, **kwargs):
    train_losses_trimmed = trimDims(train_losses) # trim to consistent dimensions
    train_losses_trimmed = np.array(train_losses_trimmed) # cast to numpy array
    train_losses_trimmed = np.transpose(train_losses_trimmed, (1,0,2)) # [run][player][hand] -> [player][run][hand]
    plotCurvesAutolabel(train_losses_trimmed, xlabel='Training Step', ylabel='MSE Loss', **kwargs)

# plot function
def plotCurvesSequence(arr_list,
            #    legend_list = None,
            #    color_list,
               x_values = [],
               upper_bound = [],
               upper_bound_label: str = 'Upper Bound',
               xlabel: str = '',
               ylabel: str = '',
               show: bool = True,
               filename: str = None,
               runSmoothing: int = 50):
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
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # set labels
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)

    # plot results
    h_list = []
    for arr in arr_list:#, legend, color in zip(arr_list, legend_list, color_list):
        x_data = range(arr.shape[1]) if len(x_values) == 0 else x_values
        numRuns = arr.shape[0]
        colors = plt.cm.jet(np.linspace(0,1,numRuns))
        for run in range(runSmoothing, numRuns+runSmoothing, runSmoothing):
            plt.plot(x_data, arr[max(0, run-runSmoothing):run].mean(axis=0), color=colors[run-runSmoothing])
        
        # compute the standard error
        # arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # # plot the mean
        # h, = ax.plot(x_data, arr.mean(axis=0), color=color, label=legend)
        # # plot the confidence band
        # arr_err = 1.96 * arr_err
        # ax.fill_between(x_data, arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3, color=color)
        # save the plot handle
        # h_list.append(h) 
    
    # plot the upper bound
    if len(upper_bound) > 0:
        x_data = range(len(upper_bound)) if len(x_values) == 0 else x_values
        h, = ax.plot(x_data, upper_bound, color='k', linestyle='--', label=upper_bound_label)
        h_list.append(h)
    
    # plot legends
    ax.legend(handles=h_list)

    # add colorbar
    colorScale = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    colorScale.set_norm(clr.Normalize(vmin=0, vmax=numRuns))
    plt.colorbar(colorScale, ax=ax, label='Game #')

    # write to file if desired
    if filename is not None:
        plt.savefig(filename)

    # show if desired
    if show:
        plt.show()
    else:
        plt.clf()

# plot training returns
def plotTrainingReturns(train_returns, **kwargs):
    train_returns_trimmed = trimDims(train_returns) # trim to consistent dimensions
    train_returns_trimmed = np.array(train_returns_trimmed) # cast to numpy array
    train_returns_trimmed = np.transpose(train_returns_trimmed, (0,2,1,3)) # [run][game][player][hand] -> [run][player][game][hand]
    numPlayers = train_returns_trimmed.shape[1]
    train_returns_trimmed = train_returns_trimmed.reshape((-1,) + train_returns_trimmed.shape[2:]) # flatten run and agent to one dimension
    roundNumbers = range(1, 60 // numPlayers + 1)
    expectedScores = [20 + 10 * (round / (0.0 + numPlayers)) for round in roundNumbers]
    plotCurvesSequence(train_returns_trimmed, x_values=roundNumbers, xlabel='Round', ylabel='Score', upper_bound=expectedScores, upper_bound_label='Nominal', **kwargs)
