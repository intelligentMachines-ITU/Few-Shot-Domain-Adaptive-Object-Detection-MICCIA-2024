import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
########################################################################################


def plot_tsne(df, x1, x2, n_classes, save_path, chunk):
    plt.figure(figsize=(16,10))
    tsne_plot = sns.scatterplot(
    x=x1, y=x2,
    palette = sns.color_palette(cc.glasbey, n_colors=n_classes), 
    data=df,
    legend="full",
    alpha=1.0
    )
    fig = tsne_plot.get_figure()
    fig.savefig(save_path+"/"+chunk)
    
    

def plot_kmeans(X,y, centers, save_path, chunk):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50)
    plt.savefig(save_path+"/"+chunk)
    

    