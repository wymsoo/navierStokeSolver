import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent / 'plots'

def plot_loss(iteration, loss):
    plt.plot(iteration,loss,label='total loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Loss plotted against iterations')
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.savefig(PLOTS_DIR / 'poiseuille_loss.png')
    plt.close()
