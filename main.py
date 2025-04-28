from lib.original_synthetic_data_generator import OriginalDataGenerator
from lib.data_distribution_analyzer import DistributionAnalyzer
from lib.sample_data_generator import SampleDataGenerator


def main():
    num_samples=10000
    random_seed=42
    save_path_original_data="./data/original_dataset.csv"
    save_path_sample_data="./data/sample_dataset.csv"

    save_path_plots="./plots/"

    ori_syn_data_gen = OriginalDataGenerator(num_samples,random_seed,save_path_original_data)
    df_original = ori_syn_data_gen.generate_and_save()
    
    # 2. Visualize the generated dataset
    analyzer = DistributionAnalyzer(df_original,save_path_plots)
    batch_test_results = analyzer.analyze()
    analyzer.print_test_summary()
    analyzer.plot_distributions()

    sample_generator = SampleDataGenerator(df_original, batch_test_results,num_samples,random_seed,save_path_plots,save_path_sample_data)
    sample_generator.generate_sample_data()
    sample_generator.plot_sample_distributions()
    sample_generator.compare_statistically()
    sample_generator.compare_visually()
    sample_generator.justify_alignment()


if __name__ == "__main__":
    main()