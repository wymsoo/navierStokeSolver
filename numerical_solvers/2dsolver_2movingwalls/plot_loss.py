import matplotlib.pyplot as plt

def plot_loss(iteration, loss):
    plt.plot(iteration[100:],loss[100:],label='total loss')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Loss plotted against iterations')
    plt.savefig('../poiseuille_loss.png')
    plt.show()
