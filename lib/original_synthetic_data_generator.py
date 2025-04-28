import numpy as np
import pandas as pd
import os

class OriginalDataGenerator:
    def __init__(self, num_samples, random_seed, save_path):
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.save_path = save_path
        # Set the random seed
        np.random.seed(self.random_seed)

    def generate_data(self):
        """
        Generate the synthetic dataset and return as a DataFrame.
        """
        df = pd.DataFrame({
            "Category1": np.random.choice(
                ["A", "B", "C", "D", "E"],
                size=self.num_samples,
                p=[0.2, 0.4, 0.2, 0.1, 0.1]
            ),
            "Value1": np.random.normal(
                loc=10, scale=2, size=self.num_samples
            ),
            "Value2": np.random.normal(
                loc=20, scale=6, size=self.num_samples
            ),
        })
        return df

    def _save_data(self, df):
        """
        Save the DataFrame to a CSV file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        df.to_csv(self.save_path, sep=";", index=False)
        print(f"Dataset saved to {self.save_path}")

    def generate_and_save(self):
        """
        Generate the data and immediately save it to CSV.
        """
        df = self.generate_data()
        print(df.head())  # Preview the first few rows
        self._save_data(df)
        return df
