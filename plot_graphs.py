import matplotlib.pyplot as plt
fp = '/Users/amanvernekar/Documents/GitHub/graph_weather/output_2022_4months_normed_singlemodel_100epochs.txt'

epochs = range(1,101)

with open(fp, 'r') as f:
    lines = f.readlines()
    train_losses = [float(line.split()[-1][:-1]) for line in lines[3:203:2]]
    val_losses = [float(line.split()[-1][:-1]) for line in lines[4:204:2]]

    plt.rcParams['figure.figsize'] = [12, 5]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Single model')
    ax1.set_title('Training loss vs epoch')
    ax2.set_title('Validation loss vs epoch')
    ax1.plot(epochs, train_losses)
    ax2.plot(epochs, val_losses)
    plt.show()
    fig.savefig('/Users/amanvernekar/Documents/GitHub/graph_weather/graph_2022_4months_normed_singlemodel_100epochs.png')