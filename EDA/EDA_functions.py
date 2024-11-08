import numpy as np
import seaborn as sns
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
from typing import List
from pydantic import BaseModel, Field, FilePath


class Config(BaseModel):
    """
    Pydantic model to define the configuration structure.
    This model ensures the configuration is valid.
    """
    columns: List[str] = Field(..., description="List of categorical columns to encode.")
    data_path: FilePath = Field(..., description="Path to the dataset CSV file.")
    drop: str  = Field(..., description="ID user we do not need")


def df_transformation_to_np(df: pl.DataFrame, config: Config) -> np.ndarray | List[str]:
    """
    Transforms a Polars DataFrame by encoding specified categorical columns into numeric values
    and returns the result as a NumPy array.

    Args:
        df (pl.DataFrame): The input Polars DataFrame containing categorical data.
        config (DictConfig): The Hydra configuration containing the column names to be encoded.

    Returns:
        np.ndarray: A NumPy array representation of the transformed DataFrame with encoded columns.
    """
    category_columns = config.columns

    # Cast categorical columns to integer codes and create new columns with the encoded values
    df = df.with_columns(
        [pl.col(col).cast(pl.Categorical).to_physical().alias(f"{col}_encoded") for col in category_columns]
    )
    df = df.drop(config.columns)
    df = df.drop(config.drop)

    return df.to_numpy(), df.columns



def correlation_matrix_calculation(df_np: np.ndarray) -> np.ndarray:
    """
    Computes the correlation matrix of a NumPy array, representing the linear relationship
    between columns (features).

    Args:
        df_np (np.ndarray): The input NumPy array where columns represent features.

    Returns:
        np.ndarray: The correlation matrix as a NumPy array.
    """

    return np.corrcoef(df_np, rowvar=False)  # rowvar=False ensures correlation is computed across columns


def plot_correlation_matrix(correlation_np: np.ndarray, labels: List[str]) -> None:
    """
    Plots the correlation matrix as a heatmap using Seaborn.

    Args:
        correlation_np (np.ndarray): The correlation matrix to be visualized.
        labels (List[str]): The labels of the columns.
    Returns:
        None: Displays the heatmap plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_np, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, xticklabels=labels, yticklabels=labels)
    plt.title("Correlation Matrix Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_violin_grid(df: pl.DataFrame, config: Config) -> None:
    """
    Plots a grid of violin plots for each numerical column in the Polars DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing the data.
        config (DictConfig): The Hydra configuration file wrapper through pydantic.

    Returns:
        None: Displays the violin plots in a grid.
    """
    # Transform the Polars Dataframe into only numerical values
    numerical_df =  df.select(cs.numeric())
    numerical_df = numerical_df .drop(config.drop)
    numerical_columns = numerical_df.columns

    # Determine the number of rows and columns for the grid
    num_columns = len(numerical_columns)
    number_of_columns = 3  # Number of plots per row
    number_of_rows = (num_columns // number_of_columns) + (1 if num_columns % number_of_columns != 0 else 0)

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, figsize=(15, 5 * number_of_rows))
    axes = axes.flatten()  # Flatten axes to easily index them

    for i, col in enumerate(numerical_columns):
        sns.violinplot(x=numerical_df[col].to_pandas(), ax = axes[i])
        axes[i].set_title(f'Violin Plot of {col}')
        axes[i].set_xlabel(col)

    # Remove empty subplots if the grid is not fully filled
    for idx in range(num_columns, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()  # Adjust layout to make sure plots are well spaced
    plt.show()


def plot_box_grid(df: pl.DataFrame, config: Config) -> None:
    """
    Plots a grid of box plots for each numerical column in the Polars DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing the data.
        config (DictConfig): The Hydra configuration file wrapper through pydantic.

    Returns:
        None: Displays the box plots in a grid.
    """
    # Transform the Polars Dataframe into only numerical values
    numerical_df = df.select(cs.numeric())
    numerical_df = numerical_df.drop(config.drop)
    numerical_columns = numerical_df.columns

    # Determine the number of rows and columns for the grid
    num_columns = len(numerical_columns)
    number_of_columns = 3  # Number of plots per row
    number_of_rows = (num_columns // number_of_columns) + (1 if num_columns % number_of_columns != 0 else 0)

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, figsize=(15, 5 * number_of_rows))
    axes = axes.flatten()  # Flatten axes to easily index them

    for i, col in enumerate(numerical_columns):
        sns.boxplot(x=numerical_df[col].to_pandas(), ax=axes[i])
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_xlabel(col)

    # Remove empty subplots if the grid is not fully filled
    for idx in range(num_columns, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()  # Adjust layout to make sure plots are well spaced
    plt.show()
def plot_histplot_grid(df: pl.DataFrame, config: Config) -> None:
    """
    Plots a grid of histogram plots for each numerical column in the Polars DataFrame.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing the data.
        config (DictConfig): The Hydra configuration file wrapper through pydantic.

    Returns:
        None: Displays the histogram plots in a grid.
    """
    # Transform the Polars Dataframe into only numerical values
    numerical_df = df.select(cs.numeric())
    numerical_df = numerical_df.drop(config.drop)
    numerical_columns = numerical_df.columns

    # Determine the number of rows and columns for the grid
    num_columns = len(numerical_columns)
    number_of_columns = 3  # Number of plots per row
    number_of_rows = (num_columns // number_of_columns) + (1 if num_columns % number_of_columns != 0 else 0)

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, figsize=(15, 5 * number_of_rows))
    axes = axes.flatten()  # Flatten axes to easily index them

    for i, col in enumerate(numerical_columns):
        sns.histplot(x=numerical_df[col].to_pandas(), ax=axes[i])
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_xlabel(col)

    # Remove empty subplots if the grid is not fully filled
    for idx in range(num_columns, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()  # Adjust layout to make sure plots are well spaced
    plt.show()


def plot_barplot_grid(df: pl.DataFrame) -> None:
    """
    Plots a grid of bar plots for each categorical column in the Polars DataFrame, with different colors for each category.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing the data.

    Returns:
        None: Displays the bar plots in a grid.
    """
    # Select only the categorical columns
    categorical_df = df.select(pl.col(pl.Utf8))  # Ensures we get only string columns
    categorical_columns = categorical_df.columns

    # Determine the number of rows and columns for the grid
    num_columns = len(categorical_columns)
    number_of_columns = 3  # Number of plots per row
    number_of_rows = (num_columns // number_of_columns) + (1 if num_columns % number_of_columns != 0 else 0)

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, figsize=(15, 5 * number_of_rows))
    axes = axes.flatten()  # Flatten axes to easily index them

    for i, col in enumerate(categorical_columns):
        # Count occurrences for each category
        value_counts = categorical_df[col].to_pandas().value_counts().reset_index()
        value_counts.columns = [col, 'count']  # Rename columns for clarity

        # Create a bar plot for each categorical column with distinct colors
        sns.barplot(x=col, y='count', data=value_counts, hue=col, ax=axes[i], palette="Set2")
        axes[i].set_title(f'Bar Plot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

        for label in axes[i].get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

    # Remove empty subplots if the grid is not fully filled
    for idx in range(num_columns, len(axes)):
        fig.delaxes(axes[idx])
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()  # Adjust layout to make sure plots are well spaced
    plt.show()

