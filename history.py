import matplotlib.pyplot as plt
import numpy as np
def plot_history(loss_list):   
    plt.figure(figsize=(12, 10))
    plt.rcParams['font.size'] = 25
    num = np.arange(1,len(loss_list)+1,1)
    print(num.shape)
    loss = np.array(loss_list)
    print(loss.shape)
    # plot loss values
    plt.plot(num, loss, color = "red", linestyle = "solid" ,label = 'train loss')
    plt.legend()
    plt.grid()    
    
    plt.savefig("history.png")
    plt.close()