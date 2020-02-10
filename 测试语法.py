import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def f1():
    x=np.arange(0,50,0.01)
    y=np.sin(x)
    plt.title("Mr zhao")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x,y,"k-o")
    plt.show()
f1()