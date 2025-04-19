import pandas as pd
import logging
from abc import ABC, abstractmethod


# Abstract Base Class for Missing Value Handling Strategy
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        pass
# Concrete Strategy: Fill a specific column with its median
class FillSpecificColumnWithMedianStrategy(MissingValueHandlingStrategy):
    def __init__(self, column: str):
        """
        Strategy to fill missing values in a specific column with its median.

        Parameters:
        column (str): The name of the column to fill.
        """
        self.column = column

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the specified column with its median.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: Updated DataFrame with missing values in the specified column filled.
        """
        logging.info(f"Filling missing values in column '{self.column}' with median.")
        df = df.copy()
        if self.column in df.columns:
            df[self.column] = df[self.column].fillna(df[self.column].median())
            logging.info(f"Missing values in '{self.column}' filled with median.")
        else:
            logging.warning(f"Column '{self.column}' not found in DataFrame.")
        return df
    


if __name__ == "__main__":

    # Example DataFrame with missing values
    df = pd.read_csv("/Users/rouablel/Uncovering-airline-passenger-satisfaction/Data/train.csv")

    print("Original DataFrame:")
    print(df.isna().sum())

    # Apply the missing value strategy
    strategy = FillSpecificColumnWithMedianStrategy(column='Arrival Delay in Minutes')
    df_filled = strategy.handle(df)

    print("\nDataFrame after applying FillSpecificColumnWithMedianStrategy:")
    print(df_filled.isna().sum())

