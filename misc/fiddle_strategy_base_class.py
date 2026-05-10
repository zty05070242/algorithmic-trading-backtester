import pandas as pd
from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    1. have a name
    2. load data via set_data()
    3. generate signals via generate_signals()
    4. return signals via get_signals()
    """
    def __init__(self, name:str="default name"):
        self.name = name
        self.data = None
        self._signals_generated = False

    def set_data(self, data:pd.DataFrame):
        self.data = data.copy()
        self._signals_generated = False

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        pass

    def get_signals(self):
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")
        if not self._signals_generated:
            raise ValueError("No data generated.")
        return self.data