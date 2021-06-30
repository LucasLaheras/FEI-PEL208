import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def transformedData2D(transData, labels, title, names):
    for label, marker, color in zip(range(3), ('>', "^", 'v'), ('blue', 'red', 'green')):
        plt.scatter(x=transData[:,0].real[labels == label], 
                    y = transData[:,1].real[labels == label],
                    marker = marker,
                    color = color,
                    alpha = 0.8,
                    label = names[label])
            
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)
    
    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom=False, top=False,  
            labelbottom=True, left=False, right=False, labelleft=True)
    
    plt.grid()
    plt.tight_layout
    plt.show()


def transformedData1D(transData, labels, title, names):
    for label, marker, color in zip(range(3), ('>', "^", 'v'), ('blue', 'red', 'green')):
        plt.scatter(x=transData[:,0].real[labels == label], 
                    y = labels[labels == label],
                    marker = marker,
                    color = color,
                    alpha = 0.8,
                    label = names[label])
            
    plt.xlabel("PC1")
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)
    
    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom=False, top=False,  
            labelbottom=True, left=False, right=False, labelleft=True)
    
    plt.grid()
    plt.tight_layout
    plt.show()