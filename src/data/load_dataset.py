# Pandas used for loading, cleaning, and handling time-series data
import pandas as pd

"""
  This function Cleans cryptocurrency OHLC data and validates it for GRU modeling.
   """
def load_dataset(data):

    # Create a copy to avoid modifying the original dataset
    data = data.copy()

    """
       Convert Date column to datetime for proper time-series operations,
       Sort data by cryptocurrency and date to preserve temporal order and
       Remove duplicate entries for the same crypto on the same date
    """

    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%y")
    data = data.sort_values(["Crypto", "Date"])

    data = data.drop_duplicates(subset=["Crypto", "Date"])

    # Validate OHLC price logic:
    # Low <= Open/Close <= High
    data = data[
        (data["Low"] <= data["Open"]) &
        (data["Open"] <= data["High"]) &
        (data["Low"] <= data["Close"]) &
        (data["Close"] <= data["High"])
        ]
    data = data.set_index("Date")

    return data




