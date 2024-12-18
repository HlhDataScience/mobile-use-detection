# TODO: Transform into a class called EdaPipeline: It includes the EdaDataExplorationConfig and all the methods related to the class. The run method must return a file with the plots.
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandera.polars as pa
import polars as pl
import polars.selectors as cs
import seaborn as sns
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, FilePath

from src.train.logging_functions.logger import setup_logging
from src.train.validation_classes.validation_configurations import (
    DataValidationConfig,
)

# CONSTANTS
LOG_FILE = Path("../../logs/EDA.log")
setup_logging(LOG_FILE)
initialize(config_path="../train/ConfTrain/Eda_config/")
HYDRA_CONFIG = compose(config_name="Eda_config")
CONFIG_DICT = OmegaConf.to_object(HYDRA_CONFIG)


class EdaDataExplorationConfig(BaseModel):
    """
    Pydantic ModelsProduction to define the configuration structure.
    This ModelsProduction ensures the configuration is valid.
    """

    columns: List[str] = Field(
        ...,
        description="List of categorical columns to encode.",
    )
    data_path: FilePath = Field(..., description="Path to the dataset CSV file.")
    drop: str = Field(..., description="ID user we do not need")


class EdaPipeline:
    def __init__(self):
        self.df_validation: DataValidationConfig = DataValidationConfig()
        self.hydra_config: DictConfig = HYDRA_CONFIG

        try:
            self.config: EdaDataExplorationConfig = EdaDataExplorationConfig(
                **self.hydra_config
            )

            logging.info("Valid eda configuration found.")
        except ValueError as e:
            logging.error(f"Failed eda configuration yalm file with {e}")
            raise e
        try:
            pl.scan_csv(self.config.original_datapath).pipe(self.df_validation.validate)

            logging.info("DataFrame Validation is correct")
        except pa.errors.SchemaError as e:
            logging.error(f"Dataframe validation failed with {e}")
            raise e

    def generate_basic_statistics(self) -> None:
        """Generate basic statistics and save to a text file."""
        pl.scan_csv(self.config.original_datapath).describe().to_pandas().to_csv(
            os.path.join(self.config.output_folder, "basic_statistics.txt"), sep="\t"
        )

    def df_transformation_to_np(
        self, return_corr_matrix: bool
    ) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
        """Transforms the DataFrame by encoding specified categorical columns into numeric values."""
        category_columns = self.config.columns

        # Cast categorical columns to integer codes and create new columns with the encoded values
        df = (
            pl.scan_csv(self.config.original_datapath)
            .with_columns(
                [
                    pl.col(col)
                    .cast(pl.Categorical)
                    .to_physical()
                    .alias(f"{col}_encoded")
                    for col in category_columns
                ]
            )
            .drop(category_columns, axis=1)
            .drop(self.config.drop, axis=1)
        )

        df_columns = df.columns
        numpy_df = df.collect().to_numpy()

        if return_corr_matrix:
            correlation_matrix = np.corrcoef(numpy_df, rowvar=False)

            return numpy_df, df_columns, correlation_matrix

        else:
            return numpy_df, df_columns

    def plot_correlation_matrix(self) -> None:
        """Plots the correlation matrix as a heatmap using Seaborn."""
        _, labels, correlation_np = self.df_transformation_to_np(
            return_corr_matrix=True
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_np,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Correlation Matrix Heatmap")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_folder, "correlation_matrix.jpg"))
        plt.close()

    def plot_violin_grid(self) -> None:
        """Plots a grid of violin plots for each numerical column in the DataFrame."""
        numerical_df = self.df.select(cs.numeric())
        numerical_df = numerical_df.drop(self.config.drop)
        numerical_columns = numerical_df.columns

        num_columns = len(numerical_columns)
        number_of_columns = 3
        number_of_rows = (num_columns // number_of_columns) + (
            1 if num_columns % number_of_columns != 0 else 0
        )

        fig, axes = plt.subplots(
            nrows=number_of_rows,
            ncols=number_of_columns,
            figsize=(15, 5 * number_of_rows),
        )
        axes = axes.flatten()

        for i, col in enumerate(numerical_columns):
            sns.violinplot(x=numerical_df[col].to_pandas(), ax=axes[i])
            axes[i].set_title(f"Violin Plot of {col}")
            axes[i].set_xlabel(col)

        for idx in range(num_columns, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_folder, "violin_plots.jpg"))
        plt.close()

    def plot_box_grid(self) -> None:
        """Plots a grid of box plots for each numerical column in the DataFrame."""
        numerical_df = self.df.select(cs.numeric())
        numerical_df = numerical_df.drop(self.config.drop)
        numerical_columns = numerical_df.columns

        num_columns = len(numerical_columns)
        number_of_columns = 3
        number_of_rows = (num_columns // number_of_columns) + (
            1 if num_columns % number_of_columns != 0 else 0
        )

        fig, axes = plt.subplots(
            nrows=number_of_rows,
            ncols=number_of_columns,
            figsize=(15, 5 * number_of_rows),
        )
        axes = axes.flatten()

        for i, col in enumerate(numerical_columns):
            sns.boxplot(x=numerical_df[col].to_pandas(), ax=axes[i])
            axes[i].set_title(f"Box Plot of {col}")
            axes[i].set_xlabel(col)

        for idx in range(num_columns, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_folder, "box_plots.jpg"))
        plt.close()

    def plot_histogram_grid(self) -> None:
        """Plots a grid of histogram plots for each numerical column in the DataFrame."""
        numerical_df = self.df.select(cs.numeric())
        numerical_df = numerical_df.drop(self.config.drop)
        numerical_columns = numerical_df.columns

        num_columns = len(numerical_columns)
        number_of_columns = 3
        number_of_rows = (num_columns // number_of_columns) + (
            1 if num_columns % number_of_columns != 0 else 0
        )

        fig, axes = plt.subplots(
            nrows=number_of_rows,
            ncols=number_of_columns,
            figsize=(15, 5 * number_of_rows),
        )
        axes = axes.flatten()

        for i, col in enumerate(numerical_columns):
            sns.histplot(numerical_df[col].to_pandas(), ax=axes[i], kde=True)
            axes[i].set_title(f"Histogram of {col}")
            axes[i].set_xlabel(col)

        for idx in range(num_columns, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_folder, "histogram_plots.jpg"))
        plt.close()

    def plot_barplot_grid(self) -> None:
        """Plots a grid of bar plots for each categorical column in the DataFrame."""
        categorical_df = self.df.select(pl.col(pl.Utf8))
        categorical_columns = categorical_df.columns

        num_columns = len(categorical_columns)
        number_of_columns = 3
        number_of_rows = (num_columns // number_of_columns) + (
            1 if num_columns % number_of_columns != 0 else 0
        )

        fig, axes = plt.subplots(
            nrows=number_of_rows,
            ncols=number_of_columns,
            figsize=(15, 5 * number_of_rows),
        )
        axes = axes.flatten()

        for i, col in enumerate(categorical_columns):
            value_counts = categorical_df[col].to_pandas().value_counts().reset_index()
            value_counts.columns = [col, "count"]

            sns.barplot(
                x=col, y="count", data=value_counts, hue=col, ax=axes[i], palette="Set2"
            )
            axes[i].set_title(f"Bar Plot of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")

            for label in axes[i].get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment("right")

        for idx in range(num_columns, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_folder, "bar_plots.jpg"))
        plt.close()

    def run(self):
        """Execute the eda pipeline."""
        os.makedirs(self.config.output_folder, exist_ok=True)
        self.load_data()
        self.generate_basic_statistics()
        _, _, correlation_matrix = self.df_transformation_to_np(return_corr_matrix=True)
        self.plot_correlation_matrix(correlation_matrix, self.df.columns)
        self.plot_violin_grid()
        self.plot_box_grid()
        self.plot_histogram_grid()
        self.plot_barplot_grid()
