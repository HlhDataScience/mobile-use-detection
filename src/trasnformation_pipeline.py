import json
import logging
import os
from typing import Any, Dict, List, Literal, Union

# import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, Field, FilePath, ValidationError

""")
from sklearn.svm import SVC
from sklearn.model_selection import(
GridSearchCV,
RandomizedSearchCV,
)
from skopt import BayesSearchCV"""

# TODO WE NEED TO CREATE A NEW JUPYTER NOTEBOOK TO TEST THE CLASS LazyTransformationPipeline & class DataTransformationConfig
# TODO WE NEED TO CHANGE THE DataValidationConfig to use pandera and see how we can integrate polars and/or pydantic
logger = logging.getLogger(__name__)
# Class Validations for Data input and data transformation


# This class needs to be changed to take advantage of lazyframe in polars as well as pandera.
class DataValidationConfig(BaseModel):
    """
    Validates the format of incoming data to ensure it conforms to the expected schema.

    Attributes:
        UserID (int): User ID.
        DeviceModel (str): Device model.
        OperatingSystem (str): Operating system used.
        AppUsageTime_min_day (int): Daily app usage time in minutes.
        ScreenOnTime_hours_day (float): Daily screen on time in hours.
        BatteryDrain_mAh_day (int): Daily battery drain in mAh.
        NumberOfAppsInstalled (int): Number of apps installed.
        DataUsage_MB_day (int): Daily data usage in MB.
        Age (int): Age in days.
        Gender (str): User's gender.
        UserBehaviorClass (int): Class label for user behavior.
    """

    UserID: int = Field(default=..., description="User ID.")
    DeviceModel: str = Field(..., description="Device Model.")
    OperatingSystem: str = Field(..., description="Operating System.")
    AppUsageTime_min_day: int = Field(..., description="App Usage Time in Minutes.")
    ScreenOnTime_hours_day: float = Field(..., description="Screen On Time in Hours.")
    BatteryDrain_mAh_day: int = Field(..., description="Battery Drain in Ah.")
    NumberOfAppsInstalled: int = Field(..., description="Number of Apps installed.")
    DataUsage_MB_day: int = Field(..., description="Data Usage in MB.")
    Age: int = Field(..., description="Age in days.")
    Gender: str = Field(..., description="Gender.")
    UserBehaviorClass: int = Field(..., description="UserBehavior Class.")

    @classmethod
    def validate_data(cls, data: Dict[str, Any]) -> bool:
        """
        Validates a dictionary of data to ensure it matches the schema.

        Args:
            data (Dict[str, Any]): Data to validate.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        try:
            cls(**data)
            return True
        except ValidationError as e:
            logger.error(f"Validation error: {e}")

    @classmethod
    def validate_csv(
        cls,
        filepath: FilePath,
        stop_on_invalid_row: bool = True,
        raise_on_invalid: bool = True,
    ) -> bool:
        """
        Validates each row of a CSV file against the schema.

        Args:
            filepath (Path): Path to the CSV file.
            stop_on_invalid_row (bool): If True, will stop processing after the first invalid row.
            raise_on_invalid (bool): If True, will raise an exception after finding an invalid row.

        Returns:
            bool: True if all rows are valid, False otherwise.

        Raises:
            ValidationErrorException: If raise_on_invalid is True and an invalid row is found.
        """
        all_valid = True

        # Use 'with' to automatically handle file opening and closing
        df = pl.read_csv(filepath)
        dfdict = df.to_dicts()
        for row in dfdict:
            if not cls.validate_data(row):
                all_valid = False
                logger.error(f"Invalid row: {row}")
                if raise_on_invalid:
                    raise TypeError(f"Invalid row found: {row}")
                if stop_on_invalid_row:
                    return False

        return all_valid

    @classmethod
    def validate_and_serialize(
        cls,
        filepath: FilePath,
        json_schema_filepath: str,
        json_schema_name: str = "json_schema.json",
    ) -> None:
        """
        Validates each row of a CSV file against the schema and serialize the file to be ready to use

        Args:
            filepath (Path): Path to the CSV file.
            json_schema_filepath (str): Path to the JSON Schema file.
            json_schema_name (str): Name of the JSON Schema file.


        Returns:
            bool: True if all rows are valid, False otherwise.
            Dict[str, Any]: Dictionary of serialized data.

        Raises:
            ValidationErrorException: If raise_on_invalid is True and an invalid row is found.
        """

        all_valid = True
        schema_path: FilePath = str(
            os.path.join(json_schema_filepath, json_schema_name)
        )

        df = pl.read_csv(filepath)
        df_dict = df.to_dicts()
        for row in df_dict:
            if not cls.validate_data(row):
                all_valid = False
                logger.error(f"Invalid row: {row}")

                raise TypeError(f"Invalid row found: {row}")

        with open(schema_path, "w") as f:
            schema_to_json = cls.model_json_schema()
            json.dump(schema_to_json, f)  # type: ignore
        print(
            f"All the files are valid : {str(all_valid)}\n\nJSON schema file saved at {schema_path}"
        )


# This class seems fine at the moment.
class DataTransformationConfig(BaseModel):
    """
    Configuration for LazyTransformationPipeline class.

    Attributes:
        original_datapath (FilePath): Path to the original data file.
        categorical_columns_to_transform (List[str]): Columns to transform from categorical to numeric.
        columns_to_drop (List[str]): Columns to drop from the DataFrame.
        normalize_df (bool): Whether to normalize the DataFrame.
        standarized_df (bool): Whether to standardize the DataFrame.
        feature_engineering_dict (Dict[str, Union[float, int, str]]): Dictionary for feature engineering.
        transformed_intermediate_df_path (FilePath): Path to the transformed intermediate data file.
        transformed_train_df_path_X (FilePath): Path to save the transformed train X DataFrame.
        transformed_train_df_path_y (FilePath): Path to save the transformed train y DataFrame.
        transformed_test_df_path_X (FilePath): Path to save the transformed test X DataFrame.
        transformed_test_df_path_y (FilePath): Path to save the transformed test y DataFrame.
        target_column (str): Name of the target column.
        feature_mode (Literal['RandomSearch', "GridSearch", "BayesianOptim"]): The feature selection mode, which can be one of 'RandomSearch', 'GridSearch', or 'BayesianOptim'.
        transformed_normalized_df_path_train_X (FilePath): Path to save the transformed normalized train X DataFrame.
        transformed_standarized_df_path_train_X (FilePath): Path to save the transformed standardized train X DataFrame.
        transformed_normalized_df_path_test_X (FilePath): Path to save the transformed normalized test X DataFrame.
        transformed_standarized_df_path_test_X (FilePath): Path to save the transformed standardized test X DataFrame.
    """

    original_datapath: FilePath = Field(
        ..., description="Path to the original data folder"
    )
    categorical_columns_to_transform: List[str] = Field(
        ...,
        description="List of columns to transform from categorical string to numerical",
    )
    columns_to_drop: List[str] = Field(..., description="List of columns to drop")
    normalize_df: bool = Field(..., description="Whether to normalize the data")
    standarized_df: bool = Field(..., description="Whether to standardize the data")
    feature_engineering_dict: Dict[str, Union[float, int, str]] = Field(
        ...,
        description="Feature engineering dictionary specifying transformations for columns",
    )
    transformed_intermediate_df_path: FilePath = Field(
        ..., description="Path to save the transformed intermediate DataFrame"
    )
    transformed_train_df_path_X: FilePath = Field(
        ..., description="Path to the transformed train data X folder"
    )
    transformed_train_df_path_y: FilePath = Field(
        ..., description="Path to the transformed train data y folder"
    )
    transformed_test_df_path_X: FilePath = Field(
        ..., description="Path to the transformed test data X folder"
    )
    transformed_test_df_path_y: FilePath = Field(
        ..., description="Path to the transformed test data Y folder"
    )
    transformed_normalized_df_path_train_X: FilePath = Field(
        ..., description="Path to save the transformed normalized train DataFrame"
    )
    transformed_standarized_df_path_train_X: FilePath = Field(
        ..., description="Path to save the transformed standardized train DataFrame"
    )
    transformed_normalized_df_path_test_X: FilePath = Field(
        ..., description="Path to save the transformed normalized test DataFrame"
    )
    transformed_standarized_df_path_test_X: FilePath = Field(
        ..., description="Path to save the transformed standardized test DataFrame"
    )
    target_column: str = Field(..., description="Name of the target column")
    feature_mode: Literal["RandomSearch", "GridSearch", "BayesianOptim"] = Field(
        ...,
        description="Feature mode selected from 'RandomSearch', 'GridSearch', or 'BayesianOptim'",
    )


# Transformation class:


# The transformation functions can be split further into a  new class Pipeline that uses some of the sklearn elements
class LazyTransformationPipeline:
    """This class leverages lazy api of polars to perform speed up transformations in dataframes using a config from pandera"""

    def __init__(self):
        self.config = DataTransformationConfig()

    def df_categorical_to_numerical(self) -> None:
        """
        Encodes categorical columns as numeric values and save the dataframe in csv format
        """
        category_columns = self.config.categorical_columns_to_transform

        # Cast categorical columns to integer codes and create new columns with the encoded values
        lf = (
            pl.scan_csv(self.config.original_datapath)
            .with_columns(
                [  # WE NEED TO CHANGE THE VALIDATION METHOD TO MAKE THIS WORK!
                    pl.col(col)
                    .cast(pl.Categorical)
                    .to_physical()
                    .alias(f"{col}_encoded")
                    for col in category_columns
                ]
            )
            .drop(self.config.categorical_columns_to_transform)
            .drop(self.config.columns_to_drop)
        )

        lf.sink_csv(self.config.transformed_intermediate_df_path)

    def split_train_test(
        self, random_state: int = 42, train_fraction: float = 0.75
    ) -> None:
        """Splits the data into train and test sets."""

        print("Loading data...")

        lazy_df = pl.scan_csv(
            self.config.transformed_intermediate_df_path
        ).with_columns(pl.all().shuffle(seed=random_state))

        print("Transforming Data...")
        train_lazy_df = lazy_df.filter(
            pl.col("row_nr") < pl.col("row_nr").max() * train_fraction
        )
        test_lazy_df = lazy_df.filter(
            pl.col("row_nr") >= pl.col("row_nr").max() * train_fraction
        )

        x_train = train_lazy_df.drop([self.config.target_column, "row_nr"])
        x_test = test_lazy_df.drop([self.config.target_column, "row_nr"])
        y_train = train_lazy_df.select(self.config.target_column)
        y_test = test_lazy_df.select(self.config.target_column)

        print("Saving Data...")
        x_train.sink_csv(self.config.transformed_train_df_path_X, index=False)
        x_test.sink_csv(self.config.transformed_test_df_path_X, index=False)
        y_train.sink_csv(self.config.transformed_train_df_path_y, index=False)
        y_test.sink_csv(self.config.transformed_test_df_path_y, index=False)

    def normalize(self) -> None:
        """performs the normalization between [0, 1] for the"""
        lazy_normalize_train_X = pl.scan_csv(
            self.config.transformed_train_df_path_X
        ).select(pl.all() - pl.all().min()) / (pl.all().max() - pl.all().min())
        lazy_normalize_test_X = pl.scan_csv(
            self.config.transformed_test_df_path_X
        ).select(pl.all() - pl.all().min()) / (pl.all().max() - pl.all().min())

        lazy_normalize_train_X.sink_csv(
            self.config.transformed_normalized_df_path_train_X
        )
        lazy_normalize_test_X.sink_csv(
            self.config.transformed_normalized_df_path_test_X
        )

    def standardize(self) -> None:
        """Performs the standardization within a normal distribution for the dataframe"""
        lazy_standardize_train_X = pl.scan_csv(
            self.config.transformed_train_df_path_X
        ).select((pl.all() - pl.all().mean()) / pl.all().std())
        lazy_standardize_test_X = pl.scan_csv(
            self.config.transformed_test_df_path_X
        ).select((pl.all() - pl.all().mean()) / pl.all().std())
        lazy_standardize_train_X.sink_csv(
            self.config.transformed_standarized_df_path_train_X
        )
        lazy_standardize_test_X.sink_csv(
            self.config.transformed_standarized_df_path_train_X
        )

    def random_search_feature_engineering(self) -> None:
        """Performs random feature engineering for the dataframe"""
        pass

    def grid_search_feature_engineering(self) -> None:
        """Performs grid search feature engineering for the dataframe"""
        pass

    def bayesian_feature_engineering(self) -> None:
        """Performs Bayesian Optimization for the dataframe"""
        pass

    def run(
        self,
    ) -> None:
        """This function runs the full pipeline."""
        print("Pipeline running...")

        self.df_categorical_to_numerical()
        self.split_train_test()
        if self.config.normalize_df:
            self.normalize()
            if self.config.feature_mode == "RandomSearch":
                self.random_search_feature_engineering()
                print("Completed random search feature engineering")
            elif self.config.feature_mode == "GridSearch":
                self.grid_search_feature_engineering()
                print("Completed grid search feature engineering")
            else:
                self.bayesian_feature_engineering()
                print("Completed bayesian feature engineering")

        elif self.config.standardize_df:
            self.standardize()
            print("Completed categorical transformation")
            if self.config.feature_mode == "RandomSearch":
                self.random_search_feature_engineering()
                print("Completed random search feature engineering")
            elif self.config.feature_mode == "GridSearch":
                self.grid_search_feature_engineering()
                print("Completed grid search feature engineering")
            else:
                self.bayesian_feature_engineering()
                print("Completed bayesian feature engineering")
        else:
            print("Completed categorical transformation")
            if self.config.feature_mode == "RandomSearch":
                self.random_search_feature_engineering()
                print("Completed random search feature engineering")
            elif self.config.feature_mode == "GridSearch":
                self.grid_search_feature_engineering()
                print("Completed grid search feature engineering")
            else:
                self.bayesian_feature_engineering()
                print("Completed bayesian feature engineering")

        return  # TODO WE NEED TO CREATE A NEW JUPYTERNOTEBOOK TO TEST THE CLASS
