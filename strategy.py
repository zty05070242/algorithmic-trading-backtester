import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict
from data_loader import load_historical_data

class Strategy(ABC):
    """
    Every strategy must be able to:
    1. Have a name
    2. Load data via set_data()
    3. Generate their own signals generate_signals()
    4. Return the signals via return_signals()
    """
    def __init__(self, name:str="default_name"):
        self.name = name
        self.data = None
        self.signals_generated = False       # tracks if generate_signals() is is called
    
    def set_data(self, data:pd.DataFrame):      # receives a pre-loaded price DataFrame and stores it into the strategy
        self.data = data.copy()
        self.signals_generated = False      # reset if new data is loaded

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        pass

    def return_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded. Call set_data() first")
        if not self.signals_generated:
            raise ValueError("Signals not yet generated. Cakk generate_signals() first.")
        return self.data
        
