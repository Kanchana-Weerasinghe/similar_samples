import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os 

class DistributionAnalyzer:
    def __init__(self, df,plot_save_path):
        self.df = df
        self.plot_save_path=plot_save_path
        self.test_results = {}
        
    def analyze(self):
        for column in self.df.columns:
            self._analyze_column(column)
        return self.test_results
    
    def _analyze_column(self, column):
        col_data = self.df[column].dropna()
        
        # Determine column type
        if pd.api.types.is_numeric_dtype(col_data):
            self._analyze_numerical(column, col_data)
        else:
            self._analyze_categorical(column, col_data)
    
    def _analyze_numerical(self, column, data):
        
        # Basic statistics
        stats = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }
        
        # Fit multiple distributions
        distributions = {
            'norm': self._fit_normal(data),
            'lognorm': self._fit_lognormal(data),
            'expon': self._fit_exponential(data),
            'gamma': self._fit_gamma(data)
        }
        
        # Select best fit using Kolmogorov-Smirnov test
        best_fit = self._select_best_fit(data, distributions)
        
        self.test_results[column] = {
            'Type': 'Numerical',
            'Statistics': stats,
            'Best Fit Distribution': best_fit,
            'All Fits': distributions
        }
    
    def _analyze_categorical(self, column, data):
        """
        Analyzes a categorical column.
        """
        value_counts = data.value_counts(normalize=True).to_dict()
        
        self.test_results[column] = {
            'Type': 'Categorical',
            'Categories': len(value_counts),
            'Value Counts': value_counts,
            'Most Common': data.mode().values[0]
        }
    
    def _fit_normal(self, data):
        """Fits a normal distribution to the data."""
        params = {
            'loc': np.mean(data),
            'scale': np.std(data)
        }
        ks_stat, p_value = stats.kstest(data, 'norm', args=(params['loc'], params['scale']))
        return {'params': params, 'ks_stat': ks_stat, 'p_value': p_value}
    
    def _fit_lognormal(self, data):
        """Fits a log-normal distribution to the data."""
        log_data = np.log(data[data > 0])
        params = {
            's': np.std(log_data),
            'scale': np.exp(np.mean(log_data))
        }
        ks_stat, p_value = stats.kstest(data, 'lognorm', args=(params['s'], 0, params['scale']))
        return {'params': params, 'ks_stat': ks_stat, 'p_value': p_value}
    
    def _fit_exponential(self, data):
        """Fits an exponential distribution to the data."""
        params = {
            'scale': 1/np.mean(data)
        }
        ks_stat, p_value = stats.kstest(data, 'expon', args=(0, 1/params['scale']))
        return {'params': params, 'ks_stat': ks_stat, 'p_value': p_value}
    
    def _fit_gamma(self, data):
        """Fits a gamma distribution to the data."""
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)
        params = {
            'shape': fit_alpha,
            'loc': fit_loc,
            'scale': fit_beta
        }
        ks_stat, p_value = stats.kstest(data, 'gamma', args=(fit_alpha, fit_loc, fit_beta))
        return {'params': params, 'ks_stat': ks_stat, 'p_value': p_value}
    
    def _select_best_fit(self, data, distributions):
        """
        Selects the best fitting distribution based on KS test p-value.
        """
        best_fit = None
        best_p = 0
        
        for name, result in distributions.items():
            if result['p_value'] > best_p:
                best_p = result['p_value']
                best_fit = {name: result['params']}
        
        return best_fit
    
    def print_test_summary(self):
        """Prints a detailed summary of distribution fitting and selection."""
        for column, results in self.test_results.items():
            print(f"\n{'='*50}")
            print(f"Column: {column}")
            print(f"Type: {results['Type']}")
            print(f"{'-'*50}")

            if results['Type'] == 'Numerical':
                print(f"Fitted Distributions and KS-Test Results for original data set:")
                for dist_name, dist_info in results['All Fits'].items():
                    ks_stat = dist_info['ks_stat']
                    p_value = dist_info['p_value']
                    print(f"  - {dist_name}: KS Statistic = {ks_stat:.4f}, p-value = {p_value:.4f}")

                best_fit_name = list(results['Best Fit Distribution'].keys())[0]
                best_p_value = results['All Fits'][best_fit_name]['p_value']
                
                print(f"\nBest Fit Distribution: {best_fit_name}")
                print(f"Reason: {best_fit_name} had the highest p-value ({best_p_value:.4f}),")
                print("indicating the best agreement with the data according to the Kolmogorov-Smirnov test.")
            
            else:  # Categorical
                print(f"Number of Categories: {results['Categories']}")
                print(f"Most Common Category: {results['Most Common']}")
            
            print(f"{'='*50}")
                
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
                    y = stats.expon.pdf(x, loc=params.get('loc', 0), scale=params['scale'])
                elif best_fit_name == 'gamma':
                    y = stats.gamma.pdf(x, a=params['shape'], loc=params['loc'], scale=params['scale'])
                
                plt.plot(x, y, 'r-', lw=2, label=f'Best fit: {best_fit_name}')
                
            else:  # Categorical
                self.df[column].value_counts(normalize=True).plot(kind='bar')
            
            plt.title(f"Distribution of {column} in original data set")
            plt.xlabel(column)
            plt.ylabel('Density' if results['Type'] == 'Numerical' else 'Proportion')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            file_path = os.path.join(self.plot_save_path, f"{column}_distribution_original_data.png")
            plt.savefig(file_path, dpi=300)
        
            plt.show()
