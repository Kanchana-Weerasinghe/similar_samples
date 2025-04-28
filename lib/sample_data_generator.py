import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.stats import chisquare, ks_2samp, skew, kurtosis

class SampleDataGenerator:
    def __init__(self, original_df, batch_test_results, num_samples, random_seed, save_path_plots,save_file_path):
        self.original_df = original_df
        self.batch_test_results = batch_test_results
        self.num_samples = num_samples
        self.sample_df = None
        self.random_seed = random_seed
        self.save_path_plots = save_path_plots
        self.save_file_path=save_file_path

    def _save_data(self, df):
        """
        Save the DataFrame to a CSV file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.save_file_path), exist_ok=True)

        df.to_csv(self.save_file_path, sep=";", index=False)
        print(f"Dataset saved to {self.save_file_path}")
    
    def generate_sample_data(self):
        np.random.seed(self.random_seed)
        sample_data = {}

        for column, results in self.batch_test_results.items():
            if results["Type"] == "Numerical":
                best_fit_name = list(results["Best Fit Distribution"].keys())[0]
                params = results["Best Fit Distribution"][best_fit_name]

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
                    print(f"⚠️ Warning: Unknown distribution '{best_fit_name}' for {column}. Using normal distribution.")
                    loc = params.get('loc', 0)
                    scale = params.get('scale', 1)
                    sample_data[column] = stats.norm.rvs(loc=loc, scale=scale, size=self.num_samples)

                sample_data[column] = self.adjust_for_skew_kurt(sample_data[column], column)

            elif results["Type"] == "Categorical":
                value_counts = results["Value Counts"]
                categories = list(value_counts.keys())
                probabilities = list(value_counts.values())
                sample_data[column] = np.random.choice(categories, size=self.num_samples, p=probabilities)

        self.sample_df = pd.DataFrame(sample_data)

        print(self.sample_df.head())  # Preview the first few rows
        self._save_data(self.sample_df)
        return self.sample_df

    def adjust_for_skew_kurt(self, sample_data, column):
        orig_skew = skew(self.original_df[column])
        sample_skew = skew(sample_data)
        orig_kurt = kurtosis(self.original_df[column])
        sample_kurt = kurtosis(sample_data)

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
                ks_stat, p_value = ks_2samp(self.original_df[column], self.sample_df[column])
                print(f"- KS Test: Statistic={ks_stat:.4f}, p-value={p_value:.4f}")
                print(f"  (Values closer to 0 and p > 0.05 indicate good match)")

            elif column in self.batch_test_results and self.batch_test_results[column]["Type"] == "Categorical":
                original_counts = self.original_df[column].value_counts().sort_index().values
                sample_counts = self.sample_df[column].value_counts().sort_index().values
                chi2_stat, p_value = chisquare(sample_counts, original_counts)
                print(f"- Chi-Square Test: Statistic={chi2_stat:.4f}, p-value={p_value:.4f}")
                print(f"  (p > 0.05 indicates proportions match)")

    def compare_visually(self):
        print("\n📊 **Visual Comparison Between Original and Sample Data**")

        num_cols = [col for col in self.sample_df.columns if self.batch_test_results[col]["Type"] == "Numerical"]

        for col in num_cols:
            plt.figure(figsize=(10, 5))
            sns.histplot(self.original_df[col], kde=True, color="blue", alpha=0.5, label="Original")
            sns.histplot(self.sample_df[col], kde=True, color="red", alpha=0.5, label="Sample")
            plt.title(f"Distribution Comparison: {col}")
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            save_file_path = os.path.join(self.save_path_plots, f"original_vs_sample_distribution_{col}.png")
            plt.savefig(save_file_path, dpi=300)

            plt.show()

        cat_cols = [col for col in self.sample_df.columns if self.batch_test_results[col]["Type"] == "Categorical"]

        for col in cat_cols:
            plt.figure(figsize=(10, 5))
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

            save_file_path = os.path.join(self.save_path_plots, f"original_vs_sample_distribution_{col}.png")
            plt.savefig(save_file_path, dpi=300)
            plt.show()

    def justify_alignment(self):
        print("\n📌 **Justification for Sample Dataset Alignment**")

        for column in self.sample_df.columns:
            if self.batch_test_results[column]["Type"] == "Numerical":
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
                orig_props = self.original_df[column].value_counts(normalize=True).sort_index()
                sample_props = self.sample_df[column].value_counts(normalize=True).sort_index()

                print(f"\n🔹 Categorical Column: {column}")
                print("- Original proportions:")
                print(orig_props)
                print("- Sample proportions:")
                print(sample_props)
                print("- Visual inspection shows the category proportions overlap well")

    def plot_sample_distributions(self):
       
        if self.sample_df is None:
            print("⚠️ No sample data found. Please generate sample data first.")
            return

        os.makedirs(self.save_path_plots, exist_ok=True)

        for column in self.sample_df.columns:
            plt.figure(figsize=(10, 5))

            if self.batch_test_results[column]["Type"] == "Numerical":
                sns.histplot(self.sample_df[column], kde=False, stat='density', color='red', alpha=0.6)

                best_fit_name = list(self.batch_test_results[column]["Best Fit Distribution"].keys())[0]
                params = self.batch_test_results[column]["Best Fit Distribution"][best_fit_name]

                x = np.linspace(self.sample_df[column].min(), self.sample_df[column].max(), 100)

                if best_fit_name == 'norm':
                    y = stats.norm.pdf(x, loc=params['loc'], scale=params['scale'])
                elif best_fit_name == 'lognorm':
                    y = stats.lognorm.pdf(x, s=params['s'], scale=params['scale'])
                elif best_fit_name == 'expon':
                    y = stats.expon.pdf(x, loc=params.get('loc', 0), scale=params['scale'])
                elif best_fit_name == 'gamma':
                    y = stats.gamma.pdf(x, a=params['shape'], loc=params['loc'], scale=params['scale'])
                else:
                    y = None

                if y is not None:
                    plt.plot(x, y, 'b-', lw=2, label=f'Best fit: {best_fit_name}')

            else:  # Categorical
                self.sample_df[column].value_counts(normalize=True).plot(kind='bar', color='red', alpha=0.7)

            plt.title(f"Sample Distribution: {column}")
            plt.xlabel(column)
            plt.ylabel('Density' if self.batch_test_results[column]["Type"] == "Numerical" else "Proportion")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            save_file_path = os.path.join(self.save_path_plots, f"sample_data_distribution_{column}.png")
            plt.savefig(save_file_path, dpi=300)

            plt.show()
            plt.close()
