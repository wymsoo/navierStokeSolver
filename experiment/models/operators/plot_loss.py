import matplotlib.pyplot as plt
from pathlib import Path



def plot_loss(iteration, loss, path_dir):
    plt.plot(iteration,loss,label='total loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Loss plotted against iterations')
    path_dir.makedirs(exist_ok=True)
    plt.savefig(path_dir / 'poiseuille_loss.png')
    plt.close()
