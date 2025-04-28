import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns



def plot_distributions(self):
    """Plots the distributions of all columns."""
    for column, results in self.test_results.items():
        plt.figure(figsize=(10, 5))
        
        if results['Type'] == 'Numerical':
            # Plot histogram and best fit
            sns.histplot(self.df[column], kde=False, stat='density')
            
            # Plot best fit curve
            best_fit_name = list(results['Best Fit Distribution'].keys())[0]
            params = results['Best Fit Distribution'][best_fit_name]
            
            x = np.linspace(self.df[column].min(), self.df[column].max(), 100)
            
            if best_fit_name == 'norm':
                y = stats.norm.pdf(x, loc=params['loc'], scale=params['scale'])
            elif best_fit_name == 'lognorm':
                y = stats.lognorm.pdf(x, s=params['s'], scale=params['scale'])
            elif best_fit_name == 'expon':
                y = stats.expon.pdf(x, loc=0, scale=1/params['scale'])  # This too could be revised for correct loc
            elif best_fit_name == 'gamma':
                # Gamma needs shape (a), loc, and scale
                y = stats.gamma.pdf(x, a=params['shape'], loc=params.get('loc', 0), scale=params['scale'])
            
            plt.plot(x, y, 'r-', lw=2, label=f'Best fit: {best_fit_name}')
            
        else:  # Categorical
            self.df[column].value_counts(normalize=True).plot(kind='bar')
        
        plt.title(f'Distribution of {column} in original data set')
        plt.xlabel(column)
        plt.ylabel('Density' if results['Type'] == 'Numerical' else 'Proportion')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
