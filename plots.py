import matplotlib.pyplot as plt

class Plots_Class:

    def printPlots(self,df_plot,label,filename=""):
                plt.clf()
                # Visualize data
                plt.scatter(
                    df_plot[df_plot.iloc[:, -1] == 0].iloc[:, 0],
                    df_plot[df_plot.iloc[:, -1] == 0].iloc[:, 1],
                    color='blue',
                    label='Class 0',
                    s=3  
                )
                plt.scatter(
                    df_plot[df_plot.iloc[:, -1] == 1].iloc[:, 0],
                    df_plot[df_plot.iloc[:, -1] == 1].iloc[:, 1],
                    color='red',
                    label='Class 1',
                    s=3  
                )
                plt.title(f'Scatter Plot of {label} Dataset')
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.legend()
                if filename!="":
                    plt.savefig(filename+"-plot.png")
                else:
                    plt.show()