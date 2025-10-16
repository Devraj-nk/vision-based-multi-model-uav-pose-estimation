import matplotlib.pyplot as plt

def plot_errors(errors):
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Position Error (m)')
    plt.title('Validation Error Over Epochs')
    plt.show()