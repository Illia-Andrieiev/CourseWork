import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class DataVisualisation:
    def plot_dataframe(self, df, x_col, main_y_col, else_y_cols=None, title='plot', x_label='X', y_label='Y', is_use_locator = False, xaxis_major_locator=365*5, yaxis_major_locator=250):
        '''
        Function for plotting using Matplotlib.

        Parameters:
        df (pd.DataFrame): Dataframe with data
        x_col (str): Column name for the X axis
        main_y_col (str): Column name for the main Y axis
        else_y_cols (list of str, optional): List of column names for additional Y axes
        title (str): Title of the graph (default 'Graph')
        x_label (str): X-axis title (default 'X')
        y_label (str): Y axis label (default 'Y')
        xaxis_major_locator (int): Step for the X-axis major grid (default 365*5 days)
        yaxis_major_locator (int): Step for the Y-axis major grid (default 250 units)
        '''
        plt.figure(figsize=(10, 6))
        plt.plot(df[x_col], df[main_y_col], marker='o', markersize=5, color='red', label=main_y_col)

        # Plot additional graphs if provided
        if else_y_cols is not None:
            for col in else_y_cols:
                plt.plot(df[x_col], df[col], marker='o', markersize=3, label=col)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, linestyle='-', linewidth=0.5)

        # Setting the Grid Density
        if is_use_locator:
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(xaxis_major_locator))  # X-axis step
            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(yaxis_major_locator))  # Y-axis step

        # Setting date format on x-axis
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Display only year
        
        # Add legend
        plt.legend()

        return plt



