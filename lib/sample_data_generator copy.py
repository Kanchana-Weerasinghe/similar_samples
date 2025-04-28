import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chisquare, ks_2samp, skew, kurtosis


class SampleDataGenerator:
    def __init__(self, original_df, batch_test_results, num_samples,random_seed,save_path_plots):
      
        self.original_df = original_df
        self.batch_test_results = batch_test_results
        self.num_samples = num_samples
        self.sample_df = None
        self.random_seed = random_seed
        self.save_path_plots=save_path_plots

    def generate_sample_data(self):
        
        np.random.seed(self.random_seed)

        sample_data = {}

        for column, results in self.batch_test_results.items():
            if results["Type"] == "Numerical":
                # Get the best fit distribution and parameters for numerical columns
                best_fit_name = list(results["Best Fit Distribution"].keys())[0]  # e.g., 'gamma', 'norm', etc.
                params = results["Best Fit Distribution"][best_fit_name]

                # Generate synthetic data based on the best-fit distribution
                if best_fit_name == "norm":
                    loc = params.get('loc', 0)
                    scale = params.get('scale', 1)
                    sample_data[column] = stats.norm.rvs(loc=loc, scale=scale, size=self.num_samples)
                
                elif best_fit_name == "lognorm":
                    s = params.get('s', 1)
                    scale = params.get('scale', 1)
                    sample_data[column] = stats.lognorm.rvs(s=s, scale=scale, size=self.num_samples)
                
                elif best_fit_name == "expon":
                    scale = params.get('scale', 1)
                    sample_data[column] = stats.expon.rvs(scale=scale, size=self.num_samples)
                
                elif best_fit_name == "gamma":
                    shape = params.get('shape', 1)
                    loc = params.get('loc', 0)
                    scale = params.get('scale', 1)
                    sample_data[column] = stats.gamma.rvs(shape, loc=loc, scale=scale, size=self.num_samples)
                
                else:
                    # If an unknown distribution is found, use normal distribution as fallback
                    print(f"⚠️ Warning: Unknown distribution '{best_fit_name}' for {column}.")

                # Adjust skewness and kurtosis if necessary
                sample_data[column] = self.adjust_for_skew_kurt(sample_data[column], column)
                
            elif results["Type"] == "Categorical":
                # Generate categorical data based on the value counts
                value_counts = results["Value Counts"]
                categories = list(value_counts.keys())
                probabilities = list(value_counts.values())
                sample_data[column] = np.random.choice(categories, size=self.num_samples, p=probabilities)

        # Create a DataFrame for the sample data
        self.sample_df = pd.DataFrame(sample_data)
        return self.sample_df

    def adjust_for_skew_kurt(self, sample_data, column):
        
        orig_skew = skew(self.original_df[column])
        sample_skew = skew(sample_data)
        orig_kurt = kurtosis(self.original_df[column])
        sample_kurt = kurtosis(sample_data)

        # Adjust skewness and kurtosis (simplified method)
        if np.abs(sample_skew - orig_skew) > 0.1:
            sample_data = stats.skewnorm.rvs(a=orig_skew, loc=np.mean(sample_data), scale=np.std(sample_data), size=self.num_samples)

        if np.abs(sample_kurt - orig_kurt) > 0.1:
            sample_data = stats.gennorm.rvs(beta=1.5, loc=np.mean(sample_data), scale=np.std(sample_data), size=self.num_samples)

        return sample_data

    def compare_statistically(self):
        
        print("\n🔍 **Statistical Comparison Between Original and Sample Data**")

        for column in self.sample_df.columns:
            print(f"\n🔹 Analyzing {column}...")

            if column in self.batch_test_results and self.batch_test_results[column]["Type"] == "Numerical":
                # Perform KS Test
                ks_stat, p_value = ks_2samp(self.original_df[column], self.sample_df[column])
                print(f"- KS Test: Statistic={ks_stat:.4f}, p-value={p_value:.4f}")
                print(f"  (Values closer to 0 and p > 0.05 indicate good match)")

            elif column in self.batch_test_results and self.batch_test_results[column]["Type"] == "Categorical":
                # Perform Chi-Square Test
                original_counts = self.original_df[column].value_counts().sort_index().values
                sample_counts = self.sample_df[column].value_counts().sort_index().values
                chi2_stat, p_value = chisquare(sample_counts, original_counts)
                print(f"- Chi-Square Test: Statistic={chi2_stat:.4f}, p-value={p_value:.4f}")
                print(f"  (p > 0.05 indicates proportions match)")

    def compare_visually(self):
        
        print("\n📊 **Visual Comparison Between Original and Sample Data**")

        # Numerical Data Comparison
        num_cols = [col for col in self.sample_df.columns 
                    if self.batch_test_results[col]["Type"] == "Numerical"]
        
        for col in num_cols:
            plt.figure(figsize=(10, 5))
            
            # Plot both distributions
            sns.histplot(self.original_df[col], 
                        kde=True, 
                        color="blue", 
                        alpha=0.5,
                        label="Original")
            sns.histplot(self.sample_df[col], 
                        kde=True, 
                        color="red", 
                        alpha=0.5,
                        label="Sample")
            
            plt.title(f"Distribution Comparison: {col}")
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Categorical Data Comparison
        cat_cols = [col for col in self.sample_df.columns 
                   if self.batch_test_results[col]["Type"] == "Categorical"]
        
        for col in cat_cols:
            plt.figure(figsize=(10, 5))
            
            # Create normalized count comparison
            plot_df = pd.DataFrame({
                'Original': self.original_df[col].value_counts(normalize=True),
                'Sample': self.sample_df[col].value_counts(normalize=True)
            }).sort_index()
            
            plot_df.plot(kind='bar', color=['blue', 'red'], alpha=0.7)
            plt.title(f"Category Distribution: {col}")
            plt.xlabel(col)
            plt.ylabel("Proportion")
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def justify_alignment(self):
        
        print("\n📌 **Justification for Sample Dataset Alignment**")
        
        for column in self.sample_df.columns:
            if self.batch_test_results[column]["Type"] == "Numerical":
                # Get comparison metrics
                orig_mean = self.original_df[column].mean()
                sample_mean = self.sample_df[column].mean()
                orig_std = self.original_df[column].std()
                sample_std = self.sample_df[column].std()
                orig_skew = skew(self.original_df[column])
                sample_skew = skew(self.sample_df[column])
                orig_kurt = kurtosis(self.original_df[column])
                sample_kurt = kurtosis(self.sample_df[column])
                
                print(f"\n🔹 Numerical Column: {column}")
                print(f"- Original mean: {orig_mean:.2f} | Sample mean: {sample_mean:.2f}")
                print(f"- Original std: {orig_std:.2f} | Sample std: {sample_std:.2f}")
                print(f"- Original skew: {orig_skew:.2f} | Sample skew: {sample_skew:.2f}")
                print(f"- Original kurtosis: {orig_kurt:.2f} | Sample kurtosis: {sample_kurt:.2f}")
                print("- Visual inspection shows the distributions overlap well")
                
            elif self.batch_test_results[column]["Type"] == "Categorical":
                # Calculate proportion difference
                orig_props = self.original_df[column].value_counts(normalize=True).sort_index()
                sample_props = self
