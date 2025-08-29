import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class CohortBenchmarkAnalysis:
    def __init__(self, file_path: str):
        """Initialize the analyzer with CSV file path"""
        self.file_path = file_path
        self.df = None
        self.cohorts = None
        self.benchmark_cohort = "Combine All"
        self.all_metrics = {}
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the CSV data"""
        print("📊 Loading and cleaning data...")
        
        # Load the CSV
        self.df = pd.read_csv(self.file_path)
        
        # Get cohort names (columns 2-14)
        self.cohorts = self.df.columns[1:].tolist()
        print(f"👥 Found {len(self.cohorts)} cohorts: {self.cohorts}")
        print(f"🎯 Using '{self.benchmark_cohort}' as benchmark")
        
        # Clean the data
        self._clean_data()
        
        return self.df
    
    def _clean_data(self):
        """Clean and standardize the data"""
        # Remove completely empty rows
        self.df = self.df.dropna(how='all')
        
        # Replace dashes with NaN
        self.df = self.df.replace('-', np.nan)
        
        # Clean percentage values and time values
        for col in self.cohorts:
            if col in self.df.columns:
                # Convert to string first
                self.df[col] = self.df[col].astype(str)
                # Remove percentage signs, time units, and clean up
                self.df[col] = self.df[col].str.replace('%', '').str.replace('secs', '').str.replace('mins', '').str.replace('sec', '')
                # Convert to numeric, errors='coerce' will convert invalid values to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def extract_all_metrics(self):
        """Extract all metrics from the dataset"""
        print("📈 Extracting all metrics...")
        
        all_metrics = {}
        
        # Process each row that contains metrics
        for idx, row in self.df.iterrows():
            metric_name = str(row.iloc[0]).strip()
            
            # Skip empty or invalid metric names
            if pd.isna(metric_name) or metric_name == '' or metric_name == 'nan':
                continue
            
            # Skip section headers
            if metric_name in ['Overall', 'Spotlight', 'Home', 'DM Dashboard', 'Bubble']:
                continue
            
            # Extract data for this metric
            metric_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(row) - 1:
                    value = row.iloc[i + 1]
                    if pd.notna(value) and value != '' and str(value) != 'nan':
                        try:
                            metric_data[cohort] = float(value)
                        except:
                            continue
            
            # Only store metrics that have data for at least one cohort
            if metric_data:
                all_metrics[metric_name] = metric_data
        
        self.all_metrics = all_metrics
        print(f"📊 Found {len(all_metrics)} metrics with data")
        return all_metrics
    
    def categorize_performance(self, metric_name: str, metric_data: Dict, benchmark_value: float):
        """Categorize cohort performance against benchmark"""
        categories = {
            'Best': {},
            'Moderate': {},
            'Low': {}
        }
        
        # Define thresholds (can be adjusted based on metric type)
        best_threshold = 1.1  # 10% better than benchmark
        low_threshold = 0.9   # 10% worse than benchmark
        
        # For time-based metrics (lower is better), reverse the logic
        time_metrics = ['time', 'efficiency', 'spent', 'pay']
        is_time_metric = any(keyword in metric_name.lower() for keyword in time_metrics)
        
        for cohort, value in metric_data.items():
            if cohort == self.benchmark_cohort:
                continue
                
            if benchmark_value == 0:
                ratio = float('inf') if value > 0 else 1
            else:
                ratio = value / benchmark_value
            
            # Categorize based on performance
            if is_time_metric:
                # For time metrics, lower is better
                if ratio <= 1/best_threshold:  # Significantly faster
                    categories['Best'][cohort] = {'value': value, 'ratio': ratio, 'performance': f'{((1-ratio)*100):+.1f}%'}
                elif ratio >= 1/low_threshold:  # Significantly slower
                    categories['Low'][cohort] = {'value': value, 'ratio': ratio, 'performance': f'{((ratio-1)*100):+.1f}%'}
                else:
                    categories['Moderate'][cohort] = {'value': value, 'ratio': ratio, 'performance': f'{((ratio-1)*100):+.1f}%'}
            else:
                # For regular metrics, higher is better
                if ratio >= best_threshold:  # Significantly better
                    categories['Best'][cohort] = {'value': value, 'ratio': ratio, 'performance': f'{((ratio-1)*100):+.1f}%'}
                elif ratio <= low_threshold:  # Significantly worse
                    categories['Low'][cohort] = {'value': value, 'ratio': ratio, 'performance': f'{((ratio-1)*100):+.1f}%'}
                else:
                    categories['Moderate'][cohort] = {'value': value, 'ratio': ratio, 'performance': f'{((ratio-1)*100):+.1f}%'}
        
        return categories
    
    def analyze_all_cohorts_performance(self):
        """Analyze all cohorts against benchmark across all metrics"""
        print("🔍 Analyzing all cohorts against benchmark...")
        
        cohort_analysis = {}
        
        # Initialize analysis for each cohort
        for cohort in self.cohorts:
            if cohort != self.benchmark_cohort:
                cohort_analysis[cohort] = {
                    'best_metrics': [],
                    'moderate_metrics': [],
                    'low_metrics': [],
                    'total_metrics': 0,
                    'performance_summary': {}
                }
        
        # Analyze each metric
        for metric_name, metric_data in self.all_metrics.items():
            # Get benchmark value
            if self.benchmark_cohort not in metric_data:
                continue
                
            benchmark_value = metric_data[self.benchmark_cohort]
            
            # Categorize performance for this metric
            categories = self.categorize_performance(metric_name, metric_data, benchmark_value)
            
            # Update cohort analysis
            for performance_level, cohorts in categories.items():
                for cohort, perf_data in cohorts.items():
                    if cohort in cohort_analysis:
                        cohort_analysis[cohort][f'{performance_level.lower()}_metrics'].append({
                            'metric': metric_name,
                            'value': perf_data['value'],
                            'benchmark': benchmark_value,
                            'ratio': perf_data['ratio'],
                            'performance': perf_data['performance']
                        })
                        cohort_analysis[cohort]['total_metrics'] += 1
        
        # Calculate performance summaries
        for cohort, analysis in cohort_analysis.items():
            total = analysis['total_metrics']
            if total > 0:
                best_count = len(analysis['best_metrics'])
                moderate_count = len(analysis['moderate_metrics'])
                low_count = len(analysis['low_metrics'])
                
                analysis['performance_summary'] = {
                    'best_percentage': (best_count / total) * 100,
                    'moderate_percentage': (moderate_count / total) * 100,
                    'low_percentage': (low_count / total) * 100,
                    'total_metrics': total
                }
        
        return cohort_analysis
    
    def print_detailed_cohort_analysis(self, cohort_analysis: Dict):
        """Print detailed analysis for each cohort"""
        print("\n" + "="*100)
        print("📊 COMPREHENSIVE COHORT ANALYSIS (vs Combine All Benchmark)")
        print("="*100)
        
        # Sort cohorts by overall performance
        sorted_cohorts = sorted(
            cohort_analysis.items(),
            key=lambda x: x[1]['performance_summary']['best_percentage'],
            reverse=True
        )
        
        for cohort, analysis in sorted_cohorts:
            if analysis['total_metrics'] == 0:
                continue
                
            summary = analysis['performance_summary']
            
            print(f"\n🏷️  {cohort.upper()} COHORT ANALYSIS")
            print("-" * 60)
            print(f"📊 Overall Performance: {summary['best_percentage']:.1f}% Best | {summary['moderate_percentage']:.1f}% Moderate | {summary['low_percentage']:.1f}% Low")
            print(f"📈 Total Metrics Analyzed: {summary['total_metrics']}")
            
            # Best performing metrics
            if analysis['best_metrics']:
                print(f"\n🏆 BEST PERFORMANCE ({len(analysis['best_metrics'])} metrics):")
                for metric_info in sorted(analysis['best_metrics'], key=lambda x: x['ratio'], reverse=True)[:5]:
                    print(f"   • {metric_info['metric']}: {metric_info['value']:.2f} vs {metric_info['benchmark']:.2f} ({metric_info['performance']})")
            
            # Low performing metrics (areas for improvement)
            if analysis['low_metrics']:
                print(f"\n⚠️  NEEDS IMPROVEMENT ({len(analysis['low_metrics'])} metrics):")
                for metric_info in sorted(analysis['low_metrics'], key=lambda x: x['ratio'])[:5]:
                    print(f"   • {metric_info['metric']}: {metric_info['value']:.2f} vs {metric_info['benchmark']:.2f} ({metric_info['performance']})")
            
            # Top opportunities
            if analysis['moderate_metrics']:
                print(f"\n📊 MODERATE PERFORMANCE ({len(analysis['moderate_metrics'])} metrics)")
        
        # Overall ranking
        print(f"\n🏆 OVERALL COHORT RANKING (by % of Best Performance):")
        for i, (cohort, analysis) in enumerate(sorted_cohorts, 1):
            if analysis['total_metrics'] > 0:
                summary = analysis['performance_summary']
                print(f"   {i}. {cohort}: {summary['best_percentage']:.1f}% best performance ({analysis['total_metrics']} metrics)")
    
    def create_performance_heatmap(self, cohort_analysis: Dict):
        """Create a performance heatmap visualization"""
        print("📊 Creating performance heatmap...")
        
        # Prepare data for heatmap
        cohorts = [c for c in cohort_analysis.keys() if cohort_analysis[c]['total_metrics'] > 0]
        
        # Get top metrics that appear across most cohorts
        metric_counts = {}
        for cohort, analysis in cohort_analysis.items():
            for metric_list in [analysis['best_metrics'], analysis['moderate_metrics'], analysis['low_metrics']]:
                for metric_info in metric_list:
                    metric_name = metric_info['metric']
                    metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
        
        # Select top metrics by frequency
        top_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        metric_names = [m[0] for m in top_metrics]
        
        # Create performance matrix
        performance_matrix = []
        cohort_labels = []
        
        for cohort in cohorts:
            analysis = cohort_analysis[cohort]
            row = []
            
            for metric_name in metric_names:
                # Find performance for this metric
                performance_score = 0  # 0 = no data, 1 = low, 2 = moderate, 3 = best
                
                for metric_info in analysis['best_metrics']:
                    if metric_info['metric'] == metric_name:
                        performance_score = 3
                        break
                
                if performance_score == 0:
                    for metric_info in analysis['moderate_metrics']:
                        if metric_info['metric'] == metric_name:
                            performance_score = 2
                            break
                
                if performance_score == 0:
                    for metric_info in analysis['low_metrics']:
                        if metric_info['metric'] == metric_name:
                            performance_score = 1
                            break
                
                row.append(performance_score)
            
            performance_matrix.append(row)
            cohort_labels.append(cohort)
        
        # Create heatmap
        plt.figure(figsize=(16, 10))
        
        # Custom colormap
        colors = ['white', 'lightcoral', 'gold', 'lightgreen']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        sns.heatmap(
            performance_matrix,
            xticklabels=[m[:30] + '...' if len(m) > 30 else m for m in metric_names],
            yticklabels=cohort_labels,
            cmap=cmap,
            vmin=0,
            vmax=3,
            annot=True,
            fmt='d',
            cbar_kws={'label': 'Performance Level (0=No Data, 1=Low, 2=Moderate, 3=Best)'}
        )
        
        plt.title('Cohort Performance Heatmap vs Combine All Benchmark', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Cohorts', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('cohort_performance_heatmap.png', dpi=300, bbox_inches='tight')
        print("📊 Performance heatmap saved as 'cohort_performance_heatmap.png'")
    
    def create_summary_dashboard(self, cohort_analysis: Dict):
        """Create a summary dashboard"""
        print("📊 Creating summary dashboard...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cohort Performance Summary Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Overall Performance Distribution
        ax1 = axes[0, 0]
        cohorts = [c for c in cohort_analysis.keys() if cohort_analysis[c]['total_metrics'] > 0]
        best_percentages = [cohort_analysis[c]['performance_summary']['best_percentage'] for c in cohorts]
        
        bars = ax1.bar(cohorts, best_percentages, color='lightgreen', alpha=0.7)
        ax1.set_title('% of Metrics Where Cohort Outperforms Benchmark', fontweight='bold')
        ax1.set_xlabel('Cohorts')
        ax1.set_ylabel('Best Performance %')
        ax1.tick_params(axis='x', rotation=45)
        plt.setp(ax1.get_xticklabels(), ha='right')
        
        for bar, value in zip(bars, best_percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. Performance Categories Distribution
        ax2 = axes[0, 1]
        categories = ['Best', 'Moderate', 'Low']
        avg_percentages = []
        
        for category in categories:
            key = f'{category.lower()}_percentage'
            values = [cohort_analysis[c]['performance_summary'][key] for c in cohorts]
            avg_percentages.append(np.mean(values))
        
        ax2.pie(avg_percentages, labels=categories, autopct='%1.1f%%', 
                colors=['lightgreen', 'gold', 'lightcoral'])
        ax2.set_title('Average Performance Distribution Across All Cohorts', fontweight='bold')
        
        # 3. Top Performers
        ax3 = axes[1, 0]
        sorted_cohorts = sorted(cohorts, key=lambda x: cohort_analysis[x]['performance_summary']['best_percentage'], reverse=True)[:8]
        top_best = [cohort_analysis[c]['performance_summary']['best_percentage'] for c in sorted_cohorts]
        
        bars = ax3.barh(sorted_cohorts, top_best, color='skyblue', alpha=0.7)
        ax3.set_title('Top Performing Cohorts (% Best Metrics)', fontweight='bold')
        ax3.set_xlabel('Best Performance %')
        
        for bar, value in zip(bars, top_best):
            width = bar.get_width()
            ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{value:.1f}%', ha='left', va='center', fontsize=9)
        
        # 4. Metrics Coverage
        ax4 = axes[1, 1]
        total_metrics = [cohort_analysis[c]['performance_summary']['total_metrics'] for c in cohorts]
        
        bars = ax4.bar(cohorts, total_metrics, color='orange', alpha=0.7)
        ax4.set_title('Number of Metrics Analyzed per Cohort', fontweight='bold')
        ax4.set_xlabel('Cohorts')
        ax4.set_ylabel('Number of Metrics')
        ax4.tick_params(axis='x', rotation=45)
        plt.setp(ax4.get_xticklabels(), ha='right')
        
        for bar, value in zip(bars, total_metrics):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(value)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('cohort_summary_dashboard.png', dpi=300, bbox_inches='tight')
        print("📊 Summary dashboard saved as 'cohort_summary_dashboard.png'")

def main():
    """Main function to run comprehensive cohort analysis"""
    print("🚀 Starting Comprehensive Cohort Analysis...")
    
    # Initialize analyzer
    analyzer = CohortBenchmarkAnalysis('Cohort Wise Analysis Fam 2.0 - Sheet1.csv')
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Extract all metrics
    all_metrics = analyzer.extract_all_metrics()
    
    # Analyze all cohorts performance
    cohort_analysis = analyzer.analyze_all_cohorts_performance()
    
    # Print detailed analysis
    analyzer.print_detailed_cohort_analysis(cohort_analysis)
    
    # Create visualizations
    analyzer.create_performance_heatmap(cohort_analysis)
    analyzer.create_summary_dashboard(cohort_analysis)
    
    print("\n✅ Comprehensive cohort analysis completed!")
    print("📊 Check the generated PNG files for visualizations")

if __name__ == "__main__":
    main()
