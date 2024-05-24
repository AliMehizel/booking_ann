import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def handle_categorical_data(df, col, orient='v'):
    """
    Visualize the distribution of categorical data.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        col (str): Name of the column containing categorical data.
        orient (str, optional): Orientation of the plot ('v' for vertical or 'h' for horizontal). Default is 'v'.
    
    Returns:
        plot: Seaborn bar plot object.
    """
    # Count occurrences of each category in the specified column
    value_counts = df[col].value_counts()
    counts_df = pd.DataFrame({col: value_counts.index, 'Count': value_counts.values})
    
    # Plot the data using Seaborn barplot
    if orient == 'v':
        plot = sns.barplot(x=counts_df[col], y=counts_df['Count'], palette='viridis', orient='v')
    elif orient == 'h':
        plot = sns.barplot(x=counts_df['Count'],y=counts_df[col], palette='viridis', orient='h')

    else:
        raise ValueError("Invalid orientation. Use 'v' for vertical or 'h' for horizontal.")
    
    return plot





def facet_grid_histogram(df, target, feature, orient='v', **kwargs):
    """
    Generate a grid of histograms for bivariate analysis.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        target (str): Column name for subsetting the data.
        feature (str): Column name for plotting.
        orient (str): Orientation of the plot. 'v' for vertical or 'h' for horizontal. Default is 'v'.
        **kwargs: Additional keyword arguments to be passed to plt.hist.
        
    Returns:
        fig: Seaborn FacetGrid object.
    """
    if orient not in ['v', 'h']:
        raise ValueError("Invalid orientation. Use 'v' for vertical or 'h' for horizontal.")

    fig = sns.FacetGrid(df, col=target)

    if orient == 'v':
        fig.map(plt.hist, feature, orient=orient, **kwargs)
    else:
        # Transpose the data to create horizontal histograms
        fig.map_dataframe(lambda data, **kwargs: plt.hist(data[feature], **kwargs, orientation='horizontal'), **kwargs)

    return fig




def compute_by(df, cols, target, by, apply_):
    """
    Compute statistics for a specific category in the target column.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        cols (list): List of column names to select.
        target (str): Column name for subsetting the data.
        by (str): Category to filter the data.
        apply_ (function): Function to apply for computation.
        
    Returns:
        res: Result after applying the function.
    """
    data = df.filter(cols, axis=1)
    res = data[data[target] == by].apply(apply_)
    
    return res


def split_dtypes(df, typ1, typ2=None, exclude=True):
    """
    Split data based on data types.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        typ1 (type or list of types): First data type(s) to include/exclude.
        typ2 (type or list of types): Second data type(s) to include/exclude.
        exclude (bool): Whether to include or exclude the specified data types.
        
    Returns:
        data_typ1, data_typ2: DataFrames with specified data types.
    """
    if exclude:
        data_typ1, data_typ2 = df.select_dtypes(include=typ1), df.select_dtypes(exclude=typ1)
    else:
        data_typ1, data_typ2 = df.select_dtypes(include=typ1), df.select_dtypes(include=typ2)
    return data_typ1, data_typ2


def handle_grid(df, plt, row, col, features, orient):
    """
    Generate complex subplot for comparison purposes.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        plt (function): Plotting function to map on the grid.
        row (str): Column name for row facet.
        col (str): Column name for column facet.
        features (list): List of features to plot.
        
    Returns:
        g: Seaborn FacetGrid object.
    """
    g = sns.FacetGrid(df, col=col, row=row, margin_titles=True)
    
    if orient not in ['v', 'h']:
        raise ValueError("Invalid orientation. Use 'v' for vertical or 'h' for horizontal.") 

    elif orient == 'h':
        g.map(plt, features, orient=orient)
    else:
        g.map(plt, features)
    return g
