import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class ComparativeAnalysis:
    def __init__(self, file_path: str, comparison_name: str, cohorts: List[str], folder_name: str):
        """Initialize comparative analysis for multiple cohorts"""
        self.file_path = file_path
        self.comparison_name = comparison_name
        self.cohorts = cohorts
        self.folder_name = folder_name
        self.benchmark_cohort = "Combine All"
        self.df = None
        self.cohorts_data = {}
        self.benchmark_data = {}
        self.comparison_results = {}
        
        # Color schemes for different comparisons
        self.color_palettes = {
            '18+ vs 18-': ['lightblue', 'lightpink'],
            'Android vs iOS': ['lightsteelblue', 'lightgray'],
            'SLS vs DM vs Bubble': ['lightyellow', 'lightpink', 'lightcyan'],
            'PPI vs TPAP vs Both': ['lightgreen', 'lightcoral', 'mediumpurple'],
            'Overall vs Ultra Users': ['orange', 'gold']
        }
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the CSV data"""
        print(f"📊 Loading data for {self.comparison_name} Analysis...")
        
        # Load the CSV
        self.df = pd.read_csv(self.file_path)
        
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
        all_cohorts = self.df.columns[1:].tolist()
        for col in all_cohorts:
            if col in self.df.columns:
                # Convert to string first
                self.df[col] = self.df[col].astype(str)
                # Remove percentage signs, time units, and clean up
                self.df[col] = self.df[col].str.replace('%', '').str.replace('secs', '').str.replace('mins', '').str.replace('sec', '')
                # Convert to numeric, errors='coerce' will convert invalid values to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def extract_cohorts_data(self):
        """Extract data for all cohorts in comparison"""
        print(f"📈 Extracting data for {', '.join(self.cohorts)}...")
        
        # Initialize data storage
        for cohort in self.cohorts:
            self.cohorts_data[cohort] = {}
        self.benchmark_data = {}
        
        # Process each row that contains metrics
        for idx, row in self.df.iterrows():
            metric_name = str(row.iloc[0]).strip()
            
            # Skip empty or invalid metric names
            if pd.isna(metric_name) or metric_name == '' or metric_name == 'nan':
                continue
            
            # Skip section headers
            if metric_name in ['Overall', 'Spotlight', 'Home', 'DM Dashboard', 'Bubble']:
                continue
            
            # Extract data for each cohort
            for cohort in self.cohorts:
                if cohort in self.df.columns:
                    cohort_value = row[cohort]
                    if pd.notna(cohort_value) and cohort_value != '' and str(cohort_value) != 'nan':
                        try:
                            self.cohorts_data[cohort][metric_name] = float(cohort_value)
                        except:
                            continue
            
            # Extract benchmark value
            if self.benchmark_cohort in self.df.columns:
                benchmark_value = row[self.benchmark_cohort]
                if pd.notna(benchmark_value) and benchmark_value != '' and str(benchmark_value) != 'nan':
                    try:
                        self.benchmark_data[metric_name] = float(benchmark_value)
                    except:
                        continue
        
        # Print data summary
        for cohort in self.cohorts:
            print(f"📊 Found {len(self.cohorts_data[cohort])} metrics for {cohort}")
        
        return self.cohorts_data, self.benchmark_data
    
    def analyze_comparative_performance(self):
        """Analyze performance differences between cohorts"""
        print(f"🔍 Analyzing comparative performance...")
        
        # Find common metrics across all cohorts
        common_metrics = set(self.cohorts_data[self.cohorts[0]].keys())
        for cohort in self.cohorts[1:]:
            common_metrics = common_metrics.intersection(set(self.cohorts_data[cohort].keys()))
        
        print(f"📊 Found {len(common_metrics)} common metrics for comparison")
        
        # Analyze each common metric
        for metric in common_metrics:
            metric_comparison = {}
            
            # Get values for each cohort
            for cohort in self.cohorts:
                cohort_value = self.cohorts_data[cohort].get(metric, 0)
                benchmark_value = self.benchmark_data.get(metric, 1)
                
                if benchmark_value != 0:
                    performance_vs_benchmark = ((cohort_value - benchmark_value) / benchmark_value) * 100
                else:
                    performance_vs_benchmark = 0
                
                metric_comparison[cohort] = {
                    'value': cohort_value,
                    'vs_benchmark': performance_vs_benchmark
                }
            
            # Calculate relative performance between cohorts
            cohort_values = [metric_comparison[cohort]['value'] for cohort in self.cohorts]
            if len(cohort_values) >= 2 and max(cohort_values) > 0:
                best_cohort = max(metric_comparison, key=lambda x: metric_comparison[x]['value'])
                worst_cohort = min(metric_comparison, key=lambda x: metric_comparison[x]['value'])
                
                best_value = metric_comparison[best_cohort]['value']
                worst_value = metric_comparison[worst_cohort]['value']
                
                if worst_value > 0:
                    performance_gap = ((best_value - worst_value) / worst_value) * 100
                else:
                    performance_gap = 0
                
                metric_comparison['analysis'] = {
                    'best_cohort': best_cohort,
                    'worst_cohort': worst_cohort,
                    'performance_gap': performance_gap,
                    'winner': best_cohort,
                    'gap_description': f"{best_cohort} outperforms {worst_cohort} by {performance_gap:.1f}%"
                }
            
            self.comparison_results[metric] = metric_comparison
        
        return self.comparison_results
    
    def get_color_palette(self):
        """Get color palette for this comparison"""
        return self.color_palettes.get(self.comparison_name, ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    
    def create_overall_comparison_viz(self):
        """Create overall comparison visualization (PNG 1)"""
        print(f"📊 Creating Overall Comparison Visualization...")
        
        colors = self.get_color_palette()
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'{self.comparison_name}: Overall Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. User Base Comparison
        ax1 = axes[0, 0]
        user_base_metric = 'Total Users'
        if user_base_metric in self.comparison_results:
            cohort_names = self.cohorts
            user_counts = [self.comparison_results[user_base_metric][cohort]['value'] for cohort in cohort_names]
            
            bars = ax1.bar(cohort_names, user_counts, color=colors[:len(cohort_names)], alpha=0.7)
            ax1.set_title('User Base Comparison', fontweight='bold')
            ax1.set_ylabel('Number of Users')
            ax1.tick_params(axis='x', rotation=45)
            plt.setp(ax1.get_xticklabels(), ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, user_counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(user_counts)*0.01,
                        f'{count:,.0f}', ha='center', va='bottom', fontsize=10)
        
        # 2. DAU Comparison
        ax2 = axes[0, 1]
        dau_metric = 'DAU (increase %)'
        if dau_metric in self.comparison_results:
            cohort_names = [cohort for cohort in self.cohorts if cohort in self.comparison_results[dau_metric]]
            dau_values = [self.comparison_results[dau_metric][cohort]['value'] for cohort in cohort_names]
            
            if cohort_names and dau_values:
                bars = ax2.bar(cohort_names, dau_values, color=colors[:len(cohort_names)], alpha=0.7)
                ax2.set_title('Daily Active Users (%)', fontweight='bold')
                ax2.set_ylabel('DAU (%)')
                ax2.tick_params(axis='x', rotation=45)
                plt.setp(ax2.get_xticklabels(), ha='right')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, dau_values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(dau_values)*0.01,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 3. Performance vs Benchmark Comparison
        ax3 = axes[1, 0]
        
        # Calculate average performance vs benchmark for each cohort
        cohort_avg_performance = {}
        for cohort in self.cohorts:
            performances = []
            for metric, data in self.comparison_results.items():
                if cohort in data and 'vs_benchmark' in data[cohort]:
                    performances.append(data[cohort]['vs_benchmark'])
            if performances:
                cohort_avg_performance[cohort] = np.mean(performances)
        
        if cohort_avg_performance:
            cohort_names = list(cohort_avg_performance.keys())
            performance_values = list(cohort_avg_performance.values())
            bar_colors = [colors[i % len(colors)] if perf > 0 else 'lightcoral' for i, perf in enumerate(performance_values)]
            
            bars = ax3.barh(cohort_names, performance_values, color=bar_colors, alpha=0.7)
            ax3.set_title('Average Performance vs Benchmark', fontweight='bold')
            ax3.set_xlabel('Performance vs Benchmark (%)')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, perf in zip(bars, performance_values):
                width = bar.get_width()
                ax3.text(width + (2 if width >= 0 else -2), bar.get_y() + bar.get_height()/2.,
                        f'{perf:+.1f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=10)
        
        # 4. Key Metrics Comparison Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create comparison summary
        summary_text = f"{self.comparison_name.upper()}\nCOMPARISON SUMMARY\n\n"
        
        # Find top performing cohort overall
        if cohort_avg_performance:
            best_overall = max(cohort_avg_performance, key=cohort_avg_performance.get)
            worst_overall = min(cohort_avg_performance, key=cohort_avg_performance.get)
            
            summary_text += f"🏆 OVERALL WINNER:\n{best_overall} ({cohort_avg_performance[best_overall]:+.1f}%)\n\n"
            summary_text += f"⚠️ NEEDS ATTENTION:\n{worst_overall} ({cohort_avg_performance[worst_overall]:+.1f}%)\n\n"
        
        # Add key insights
        summary_text += "📊 KEY INSIGHTS:\n"
        
        # Find biggest performance gaps
        performance_gaps = []
        for metric, data in self.comparison_results.items():
            if 'analysis' in data and data['analysis']['performance_gap'] > 10:
                performance_gaps.append({
                    'metric': metric,
                    'gap': data['analysis']['performance_gap'],
                    'winner': data['analysis']['winner'],
                    'description': data['analysis']['gap_description']
                })
        
        # Sort by biggest gaps
        performance_gaps.sort(key=lambda x: x['gap'], reverse=True)
        
        for gap in performance_gaps[:3]:  # Top 3 gaps
            summary_text += f"• {gap['metric'][:25]}...\n  {gap['description'][:40]}...\n\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{self.folder_name}/01_overall_comparison.png', dpi=300, bbox_inches='tight')
        print(f"📊 Overall comparison visualization saved")
    
    def create_feature_comparison_viz(self):
        """Create feature-wise comparison visualization (PNG 2)"""
        print(f"🔍 Creating Feature Comparison Visualization...")
        
        colors = self.get_color_palette()
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'{self.comparison_name}: Feature Performance Comparison', fontsize=16, fontweight='bold')
        
        # Define feature categories
        feature_categories = {
            'Spotlight Features': [
                'users opening spotlight / total users coming to home',
                'Avg. time from input entered → payment compose (search efficiency)',
                '% of SS sessions with quick actions usage'
            ],
            'DM Features': [
                'avg dm session per day',
                'Messages sent per session (to understand engagement depth)',
                '% of DM users using pins at least once (adoption)'
            ],
            'Scanning Features': [
                'user scanning / total users coming to home (user wise)',
                'Full Screen Scanner / Home Screen Scanner'
            ],
            'Payment Features': [
                'DTU',
                'time to pay per user per pay session',
                'Recent Ticket Size'
            ]
        }
        
        # 1. Spotlight Features Comparison
        ax1 = axes[0, 0]
        spotlight_metrics = [m for m in feature_categories['Spotlight Features'] if m in self.comparison_results]
        if spotlight_metrics:
            metric_names = [m.split('/')[-1].strip()[:15] for m in spotlight_metrics[:3]]  # Shortened names
            
            x = np.arange(len(metric_names))
            width = 0.35 if len(self.cohorts) == 2 else 0.25
            
            for i, cohort in enumerate(self.cohorts):
                values = []
                for metric in spotlight_metrics[:3]:
                    if cohort in self.comparison_results[metric]:
                        values.append(self.comparison_results[metric][cohort]['value'])
                    else:
                        values.append(0)
                
                offset = (i - len(self.cohorts)/2 + 0.5) * width
                bars = ax1.bar(x + offset, values, width, label=cohort, 
                              color=colors[i % len(colors)], alpha=0.7)
            
            ax1.set_title('Spotlight Features Comparison', fontweight='bold')
            ax1.set_ylabel('Values')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metric_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. DM Features Comparison
        ax2 = axes[0, 1]
        dm_metrics = [m for m in feature_categories['DM Features'] if m in self.comparison_results]
        if dm_metrics:
            metric_names = [m.split('/')[-1].strip()[:15] for m in dm_metrics[:3]]
            
            x = np.arange(len(metric_names))
            
            for i, cohort in enumerate(self.cohorts):
                values = []
                for metric in dm_metrics[:3]:
                    if cohort in self.comparison_results[metric]:
                        values.append(self.comparison_results[metric][cohort]['value'])
                    else:
                        values.append(0)
                
                offset = (i - len(self.cohorts)/2 + 0.5) * width
                bars = ax2.bar(x + offset, values, width, label=cohort, 
                              color=colors[i % len(colors)], alpha=0.7)
            
            ax2.set_title('DM Features Comparison', fontweight='bold')
            ax2.set_ylabel('Values')
            ax2.set_xticks(x)
            ax2.set_xticklabels(metric_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Performance Gaps Heatmap
        ax3 = axes[1, 0]
        
        # Create performance matrix for heatmap
        performance_metrics = []
        performance_matrix = []
        
        # Select top metrics with biggest gaps
        top_gap_metrics = []
        for metric, data in self.comparison_results.items():
            if 'analysis' in data and data['analysis']['performance_gap'] > 5:
                top_gap_metrics.append((metric, data['analysis']['performance_gap']))
        
        top_gap_metrics.sort(key=lambda x: x[1], reverse=True)
        selected_metrics = [m[0] for m in top_gap_metrics[:8]]  # Top 8 metrics
        
        if selected_metrics:
            for metric in selected_metrics:
                row = []
                for cohort in self.cohorts:
                    if cohort in self.comparison_results[metric]:
                        # Normalize performance vs benchmark
                        perf = self.comparison_results[metric][cohort]['vs_benchmark']
                        row.append(perf)
                    else:
                        row.append(0)
                performance_matrix.append(row)
                performance_metrics.append(metric[:20] + '...' if len(metric) > 20 else metric)
            
            # Create heatmap
            im = ax3.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
            ax3.set_title('Performance vs Benchmark Heatmap', fontweight='bold')
            ax3.set_xticks(range(len(self.cohorts)))
            ax3.set_xticklabels(self.cohorts)
            ax3.set_yticks(range(len(performance_metrics)))
            ax3.set_yticklabels(performance_metrics)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label('Performance vs Benchmark (%)')
            
            # Add text annotations
            for i in range(len(performance_metrics)):
                for j in range(len(self.cohorts)):
                    text = ax3.text(j, i, f'{performance_matrix[i][j]:.0f}%',
                                   ha="center", va="center", color="black", fontsize=8)
        
        # 4. Winner Analysis
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Count wins for each cohort
        cohort_wins = {cohort: 0 for cohort in self.cohorts}
        total_comparisons = 0
        
        for metric, data in self.comparison_results.items():
            if 'analysis' in data:
                winner = data['analysis']['winner']
                if winner in cohort_wins:
                    cohort_wins[winner] += 1
                total_comparisons += 1
        
        winner_text = f"{self.comparison_name.upper()}\nWINNER ANALYSIS\n\n"
        
        # Sort cohorts by wins
        sorted_winners = sorted(cohort_wins.items(), key=lambda x: x[1], reverse=True)
        
        winner_text += "🏆 FEATURE WINS LEADERBOARD:\n"
        for i, (cohort, wins) in enumerate(sorted_winners, 1):
            win_percentage = (wins / total_comparisons * 100) if total_comparisons > 0 else 0
            winner_text += f"{i}. {cohort}: {wins}/{total_comparisons} ({win_percentage:.1f}%)\n"
        
        winner_text += f"\n📊 ANALYSIS:\n"
        if sorted_winners:
            champion = sorted_winners[0]
            winner_text += f"• {champion[0]} dominates with {champion[1]} feature wins\n"
            
            if len(sorted_winners) > 1:
                runner_up = sorted_winners[1]
                gap = champion[1] - runner_up[1]
                winner_text += f"• {gap} feature advantage over {runner_up[0]}\n"
        
        ax4.text(0.05, 0.95, winner_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{self.folder_name}/02_feature_comparison.png', dpi=300, bbox_inches='tight')
        print(f"🔍 Feature comparison visualization saved")
    
    def create_engagement_analysis_viz(self):
        """Create engagement analysis visualization (PNG 3)"""
        print(f"💬 Creating Engagement Analysis Visualization...")
        
        colors = self.get_color_palette()
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'{self.comparison_name}: Engagement & Usage Analysis', fontsize=16, fontweight='bold')
        
        # 1. Session Engagement Comparison
        ax1 = axes[0, 0]
        engagement_metrics = [
            'Avg Time Spent per session',
            'avg dm session per day',
            '% of DM sessions with repeat opens in the same day (re-engagement within a day)'
        ]
        
        available_engagement = [m for m in engagement_metrics if m in self.comparison_results]
        if available_engagement:
            # Create radar chart for engagement metrics
            angles = np.linspace(0, 2 * np.pi, len(available_engagement), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for i, cohort in enumerate(self.cohorts):
                values = []
                for metric in available_engagement:
                    if cohort in self.comparison_results[metric]:
                        # Normalize values for radar chart
                        value = self.comparison_results[metric][cohort]['value']
                        benchmark = self.benchmark_data.get(metric, 1)
                        normalized_value = (value / benchmark * 100) if benchmark > 0 else 100
                        values.append(normalized_value)
                    else:
                        values.append(100)  # Baseline
                
                values += values[:1]  # Complete the circle
                
                ax1 = plt.subplot(2, 2, 1, projection='polar')
                ax1.plot(angles, values, 'o-', linewidth=2, label=cohort, color=colors[i % len(colors)])
                ax1.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
            
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels([m[:15] + '...' if len(m) > 15 else m for m in available_engagement])
            ax1.set_title('Engagement Metrics Radar\n(vs Benchmark)', fontweight='bold', pad=20)
            ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 2. Usage Intensity Comparison
        ax2 = axes[0, 1]
        usage_metrics = [
            'user scanning / total users coming to home (user wise)',
            'Messages sent per session (to understand engagement depth)',
            '% of SS sessions with quick actions usage'
        ]
        
        available_usage = [m for m in usage_metrics if m in self.comparison_results]
        if available_usage:
            metric_names = [m.split('/')[-1].strip()[:20] for m in available_usage]
            
            x = np.arange(len(metric_names))
            width = 0.35 if len(self.cohorts) == 2 else 0.25
            
            for i, cohort in enumerate(self.cohorts):
                values = []
                for metric in available_usage:
                    if cohort in self.comparison_results[metric]:
                        values.append(self.comparison_results[metric][cohort]['value'])
                    else:
                        values.append(0)
                
                offset = (i - len(self.cohorts)/2 + 0.5) * width
                bars = ax2.bar(x + offset, values, width, label=cohort, 
                              color=colors[i % len(colors)], alpha=0.7)
            
            ax2.set_title('Usage Intensity Comparison', fontweight='bold')
            ax2.set_ylabel('Values')
            ax2.set_xticks(x)
            ax2.set_xticklabels(metric_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Adoption Rates Comparison
        ax3 = axes[1, 0]
        adoption_metrics = []
        for metric in self.comparison_results.keys():
            if 'adoption' in metric.lower() or ('% of' in metric and 'using' in metric):
                adoption_metrics.append(metric)
        
        if adoption_metrics[:4]:  # Top 4 adoption metrics
            adoption_data = {}
            for cohort in self.cohorts:
                adoption_data[cohort] = []
                for metric in adoption_metrics[:4]:
                    if cohort in self.comparison_results[metric]:
                        adoption_data[cohort].append(self.comparison_results[metric][cohort]['value'])
                    else:
                        adoption_data[cohort].append(0)
            
            # Create grouped bar chart
            metric_names = [m[:15] + '...' if len(m) > 15 else m for m in adoption_metrics[:4]]
            x = np.arange(len(metric_names))
            width = 0.35 if len(self.cohorts) == 2 else 0.25
            
            for i, cohort in enumerate(self.cohorts):
                offset = (i - len(self.cohorts)/2 + 0.5) * width
                bars = ax3.bar(x + offset, adoption_data[cohort], width, label=cohort, 
                              color=colors[i % len(colors)], alpha=0.7)
            
            ax3.set_title('Feature Adoption Rates (%)', fontweight='bold')
            ax3.set_ylabel('Adoption Rate (%)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(metric_names, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Engagement Quality Score
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate engagement quality score for each cohort
        engagement_scores = {}
        quality_metrics = [
            'Avg Time Spent per session',
            'Messages sent per session (to understand engagement depth)',
            '% of DM sessions with repeat opens in the same day (re-engagement within a day)',
            'user scanning / total users coming to home (user wise)'
        ]
        
        for cohort in self.cohorts:
            scores = []
            for metric in quality_metrics:
                if metric in self.comparison_results and cohort in self.comparison_results[metric]:
                    # Score based on performance vs benchmark
                    vs_benchmark = self.comparison_results[metric][cohort]['vs_benchmark']
                    # Convert to 0-100 scale (0 = -50% vs benchmark, 100 = +50% vs benchmark)
                    score = max(0, min(100, 50 + vs_benchmark))
                    scores.append(score)
            
            if scores:
                engagement_scores[cohort] = np.mean(scores)
        
        quality_text = f"{self.comparison_name.upper()}\nENGAGEMENT QUALITY ANALYSIS\n\n"
        
        if engagement_scores:
            # Sort by engagement score
            sorted_scores = sorted(engagement_scores.items(), key=lambda x: x[1], reverse=True)
            
            quality_text += "🏆 ENGAGEMENT QUALITY RANKING:\n"
            for i, (cohort, score) in enumerate(sorted_scores, 1):
                quality_text += f"{i}. {cohort}: {score:.1f}/100\n"
            
            quality_text += f"\n📊 INSIGHTS:\n"
            if len(sorted_scores) >= 2:
                best = sorted_scores[0]
                worst = sorted_scores[-1]
                gap = best[1] - worst[1]
                quality_text += f"• {best[0]} leads in engagement quality\n"
                quality_text += f"• {gap:.1f} point gap vs {worst[0]}\n"
                
                if gap > 20:
                    quality_text += f"• Significant engagement disparity\n"
                elif gap > 10:
                    quality_text += f"• Moderate engagement difference\n"
                else:
                    quality_text += f"• Similar engagement levels\n"
        
        ax4.text(0.05, 0.95, quality_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{self.folder_name}/03_engagement_analysis.png', dpi=300, bbox_inches='tight')
        print(f"💬 Engagement analysis visualization saved")
    
    def create_performance_gaps_viz(self):
        """Create performance gaps analysis visualization (PNG 4)"""
        print(f"📊 Creating Performance Gaps Analysis Visualization...")
        
        colors = self.get_color_palette()
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'{self.comparison_name}: Performance Gaps & Opportunities', fontsize=16, fontweight='bold')
        
        # 1. Biggest Performance Gaps
        ax1 = axes[0, 0]
        
        # Find metrics with biggest performance gaps
        gap_analysis = []
        for metric, data in self.comparison_results.items():
            if 'analysis' in data:
                gap_analysis.append({
                    'metric': metric,
                    'gap': data['analysis']['performance_gap'],
                    'winner': data['analysis']['winner'],
                    'loser': data['analysis']['worst_cohort']
                })
        
        gap_analysis.sort(key=lambda x: x['gap'], reverse=True)
        top_gaps = gap_analysis[:8]  # Top 8 gaps
        
        if top_gaps:
            metric_names = [gap['metric'][:25] + '...' if len(gap['metric']) > 25 else gap['metric'] for gap in top_gaps]
            gap_values = [gap['gap'] for gap in top_gaps]
            
            bars = ax1.barh(metric_names, gap_values, color='lightcoral', alpha=0.7)
            ax1.set_title('Biggest Performance Gaps', fontweight='bold')
            ax1.set_xlabel('Performance Gap (%)')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, gap in zip(bars, gap_values):
                width = bar.get_width()
                ax1.text(width + gap * 0.01, bar.get_y() + bar.get_height()/2.,
                        f'{gap:.1f}%', ha='left', va='center', fontsize=9)
        
        # 2. Win/Loss Matrix
        ax2 = axes[0, 1]
        
        # Create win/loss matrix
        win_matrix = np.zeros((len(self.cohorts), len(self.cohorts)))
        cohort_indices = {cohort: i for i, cohort in enumerate(self.cohorts)}
        
        for metric, data in self.comparison_results.items():
            if 'analysis' in data:
                winner_idx = cohort_indices.get(data['analysis']['winner'])
                loser_idx = cohort_indices.get(data['analysis']['worst_cohort'])
                if winner_idx is not None and loser_idx is not None:
                    win_matrix[winner_idx][loser_idx] += 1
        
        im = ax2.imshow(win_matrix, cmap='Greens', aspect='auto')
        ax2.set_title('Win/Loss Matrix', fontweight='bold')
        ax2.set_xticks(range(len(self.cohorts)))
        ax2.set_xticklabels(self.cohorts, rotation=45, ha='right')
        ax2.set_yticks(range(len(self.cohorts)))
        ax2.set_yticklabels(self.cohorts)
        ax2.set_xlabel('Loses to →')
        ax2.set_ylabel('← Wins against')
        
        # Add text annotations
        for i in range(len(self.cohorts)):
            for j in range(len(self.cohorts)):
                if i != j:  # Don't show diagonal
                    text = ax2.text(j, i, f'{int(win_matrix[i][j])}',
                                   ha="center", va="center", color="black", fontweight='bold')
        
        # 3. Opportunity Analysis
        ax3 = axes[1, 0]
        
        # Find where each cohort has the biggest opportunities (worst performance gaps)
        cohort_opportunities = {cohort: [] for cohort in self.cohorts}
        
        for metric, data in self.comparison_results.items():
            if 'analysis' in data and data['analysis']['performance_gap'] > 10:
                loser = data['analysis']['worst_cohort']
                if loser in cohort_opportunities:
                    cohort_opportunities[loser].append({
                        'metric': metric,
                        'gap': data['analysis']['performance_gap'],
                        'vs_cohort': data['analysis']['winner']
                    })
        
        # Count opportunities by cohort
        opportunity_counts = {cohort: len(opps) for cohort, opps in cohort_opportunities.items()}
        
        if opportunity_counts:
            cohort_names = list(opportunity_counts.keys())
            opp_counts = list(opportunity_counts.values())
            
            bars = ax3.bar(cohort_names, opp_counts, color='orange', alpha=0.7)
            ax3.set_title('Improvement Opportunities by Cohort', fontweight='bold')
            ax3.set_ylabel('Number of Major Gaps (>10%)')
            ax3.tick_params(axis='x', rotation=45)
            plt.setp(ax3.get_xticklabels(), ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, opp_counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(count)}', ha='center', va='bottom', fontsize=10)
        
        # 4. Strategic Recommendations
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        recommendations_text = f"{self.comparison_name.upper()}\nSTRATEGIC RECOMMENDATIONS\n\n"
        
        # Find overall winner and biggest opportunities
        if gap_analysis:
            # Most dominant cohort
            winner_counts = {}
            for gap in gap_analysis:
                winner = gap['winner']
                winner_counts[winner] = winner_counts.get(winner, 0) + 1
            
            if winner_counts:
                top_performer = max(winner_counts, key=winner_counts.get)
                recommendations_text += f"🏆 LEVERAGE SUCCESS:\n{top_performer} dominates {winner_counts[top_performer]} metrics\n\n"
            
            # Biggest single opportunity
            if gap_analysis[0]['gap'] > 20:
                biggest_gap = gap_analysis[0]
                recommendations_text += f"🎯 PRIORITY OPPORTUNITY:\n{biggest_gap['loser']} in {biggest_gap['metric'][:30]}...\n{biggest_gap['gap']:.1f}% gap vs {biggest_gap['winner']}\n\n"
            
            # Strategic focus areas
            recommendations_text += "💡 STRATEGIC FOCUS:\n"
            
            if len(self.cohorts) == 2:
                # Head-to-head comparison
                cohort1, cohort2 = self.cohorts
                cohort1_wins = sum(1 for gap in gap_analysis if gap['winner'] == cohort1)
                cohort2_wins = sum(1 for gap in gap_analysis if gap['winner'] == cohort2)
                
                if cohort1_wins > cohort2_wins:
                    recommendations_text += f"• Scale {cohort1} success patterns\n"
                    recommendations_text += f"• Address {cohort2} performance gaps\n"
                else:
                    recommendations_text += f"• Scale {cohort2} success patterns\n"
                    recommendations_text += f"• Address {cohort1} performance gaps\n"
            else:
                # Multi-cohort comparison
                recommendations_text += f"• Focus on top performer patterns\n"
                recommendations_text += f"• Address systematic gaps\n"
                recommendations_text += f"• Optimize resource allocation\n"
        
        ax4.text(0.05, 0.95, recommendations_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{self.folder_name}/04_performance_gaps.png', dpi=300, bbox_inches='tight')
        print(f"📊 Performance gaps visualization saved")
    
    def create_comprehensive_summary_viz(self):
        """Create comprehensive summary dashboard (PNG 5)"""
        print(f"📊 Creating Comprehensive Summary Dashboard...")
        
        colors = self.get_color_palette()
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'{self.comparison_name}: Comprehensive Analysis Summary', fontsize=18, fontweight='bold')
        
        # 1. Overall Winner Circle
        ax1 = axes[0, 0]
        
        # Calculate overall scores for each cohort
        cohort_scores = {}
        for cohort in self.cohorts:
            scores = []
            for metric, data in self.comparison_results.items():
                if cohort in data and 'vs_benchmark' in data[cohort]:
                    scores.append(data[cohort]['vs_benchmark'])
            if scores:
                cohort_scores[cohort] = np.mean(scores)
        
        if cohort_scores:
            # Create pie chart of relative performance
            cohort_names = list(cohort_scores.keys())
            # Normalize scores to positive values for pie chart
            min_score = min(cohort_scores.values())
            normalized_scores = [score - min_score + 1 for score in cohort_scores.values()]
            
            wedges, texts, autotexts = ax1.pie(normalized_scores, labels=cohort_names, 
                                              colors=colors[:len(cohort_names)], autopct='%1.1f%%', startangle=90)
            ax1.set_title('Overall Performance Share', fontweight='bold')
        
        # 2. Performance Timeline/Trend
        ax2 = axes[0, 1]
        
        # Create performance comparison across key metrics
        key_metrics = ['DAU (increase %)', 'DTU', 'Avg Time Spent per session', 'user scanning / total users coming to home (user wise)']
        available_key_metrics = [m for m in key_metrics if m in self.comparison_results]
        
        if available_key_metrics:
            for i, cohort in enumerate(self.cohorts):
                values = []
                for metric in available_key_metrics:
                    if cohort in self.comparison_results[metric]:
                        # Normalize to percentage vs benchmark
                        vs_benchmark = self.comparison_results[metric][cohort]['vs_benchmark']
                        values.append(vs_benchmark)
                    else:
                        values.append(0)
                
                ax2.plot(range(len(available_key_metrics)), values, 'o-', 
                        linewidth=2, markersize=8, label=cohort, color=colors[i % len(colors)])
            
            ax2.set_title('Key Metrics Performance Trend', fontweight='bold')
            ax2.set_ylabel('Performance vs Benchmark (%)')
            ax2.set_xticks(range(len(available_key_metrics)))
            ax2.set_xticklabels([m[:15] + '...' for m in available_key_metrics], rotation=45, ha='right')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Champion Analysis
        ax3 = axes[0, 2]
        ax3.axis('off')
        
        champion_text = f"{self.comparison_name.upper()}\nCHAMPION ANALYSIS\n\n"
        
        if cohort_scores:
            champion = max(cohort_scores, key=cohort_scores.get)
            runner_up = sorted(cohort_scores.items(), key=lambda x: x[1], reverse=True)[1] if len(cohort_scores) > 1 else None
            
            champion_text += f"🏆 OVERALL CHAMPION:\n{champion}\n"
            champion_text += f"Score: {cohort_scores[champion]:+.1f}%\n\n"
            
            if runner_up:
                gap = cohort_scores[champion] - runner_up[1]
                champion_text += f"🥈 RUNNER-UP:\n{runner_up[0]}\n"
                champion_text += f"Gap: {gap:.1f} percentage points\n\n"
            
            # Find champion's strongest areas
            champion_strengths = []
            for metric, data in self.comparison_results.items():
                if 'analysis' in data and data['analysis']['winner'] == champion:
                    champion_strengths.append((metric, data['analysis']['performance_gap']))
            
            champion_strengths.sort(key=lambda x: x[1], reverse=True)
            
            if champion_strengths:
                champion_text += f"💪 {champion.upper()} DOMINATES IN:\n"
                for metric, gap in champion_strengths[:3]:
                    champion_text += f"• {metric[:25]}...\n  +{gap:.1f}% advantage\n"
        
        ax3.text(0.05, 0.95, champion_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=colors[0], alpha=0.2))
        
        # 4. Weakness Analysis
        ax4 = axes[1, 0]
        ax4.axis('off')
        
        weakness_text = f"⚠️ WEAKNESS ANALYSIS\n\n"
        
        if cohort_scores:
            weakest = min(cohort_scores, key=cohort_scores.get)
            weakness_text += f"🔴 NEEDS MOST HELP:\n{weakest}\n"
            weakness_text += f"Score: {cohort_scores[weakest]:+.1f}%\n\n"
            
            # Find weakest cohort's biggest gaps
            weakest_gaps = []
            for metric, data in self.comparison_results.items():
                if 'analysis' in data and data['analysis']['worst_cohort'] == weakest:
                    weakest_gaps.append((metric, data['analysis']['performance_gap']))
            
            weakest_gaps.sort(key=lambda x: x[1], reverse=True)
            
            if weakest_gaps:
                weakness_text += f"📉 {weakest.upper()} STRUGGLES WITH:\n"
                for metric, gap in weakest_gaps[:3]:
                    weakness_text += f"• {metric[:25]}...\n  -{gap:.1f}% behind leader\n"
        
        ax4.text(0.05, 0.95, weakness_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        
        # 5. Metric Categories Performance
        ax5 = axes[1, 1]
        
        # Group metrics by categories and compare
        metric_categories = {
            'Engagement': ['DAU', 'DTU', 'Avg Time Spent', 'dm session'],
            'Features': ['spotlight', 'scanning', 'adoption', 'usage'],
            'Payments': ['pay', 'transaction', 'Ticket Size', 'UPI']
        }
        
        category_performance = {cohort: {} for cohort in self.cohorts}
        
        for category, keywords in metric_categories.items():
            for cohort in self.cohorts:
                category_scores = []
                for metric, data in self.comparison_results.items():
                    if any(keyword.lower() in metric.lower() for keyword in keywords):
                        if cohort in data and 'vs_benchmark' in data[cohort]:
                            category_scores.append(data[cohort]['vs_benchmark'])
                
                if category_scores:
                    category_performance[cohort][category] = np.mean(category_scores)
        
        # Create grouped bar chart
        categories = list(metric_categories.keys())
        x = np.arange(len(categories))
        width = 0.35 if len(self.cohorts) == 2 else 0.25
        
        for i, cohort in enumerate(self.cohorts):
            values = [category_performance[cohort].get(cat, 0) for cat in categories]
            offset = (i - len(self.cohorts)/2 + 0.5) * width
            bars = ax5.bar(x + offset, values, width, label=cohort, 
                          color=colors[i % len(colors)], alpha=0.7)
        
        ax5.set_title('Performance by Category', fontweight='bold')
        ax5.set_ylabel('Avg Performance vs Benchmark (%)')
        ax5.set_xticks(x)
        ax5.set_xticklabels(categories)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. User Base vs Performance
        ax6 = axes[1, 2]
        
        # Scatter plot of user base vs performance
        if 'Total Users' in self.comparison_results:
            user_bases = []
            performances = []
            cohort_labels = []
            
            for cohort in self.cohorts:
                if cohort in self.comparison_results['Total Users'] and cohort in cohort_scores:
                    user_base = self.comparison_results['Total Users'][cohort]['value']
                    performance = cohort_scores[cohort]
                    user_bases.append(user_base)
                    performances.append(performance)
                    cohort_labels.append(cohort)
            
            if user_bases and performances:
                scatter = ax6.scatter(user_bases, performances, 
                                    c=colors[:len(cohort_labels)], s=200, alpha=0.7)
                
                # Add labels
                for i, label in enumerate(cohort_labels):
                    ax6.annotate(label, (user_bases[i], performances[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=10)
                
                ax6.set_title('User Base vs Performance', fontweight='bold')
                ax6.set_xlabel('User Base')
                ax6.set_ylabel('Performance Score (%)')
                ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax6.grid(True, alpha=0.3)
        
        # 7. Business Impact Assessment
        ax7 = axes[2, 0]
        ax7.axis('off')
        
        # Recalculate gap_analysis for this method
        gap_analysis = []
        for metric, data in self.comparison_results.items():
            if 'analysis' in data:
                gap_analysis.append({
                    'metric': metric,
                    'gap': data['analysis']['performance_gap'],
                    'winner': data['analysis']['winner'],
                    'loser': data['analysis']['worst_cohort']
                })
        gap_analysis.sort(key=lambda x: x['gap'], reverse=True)
        
        impact_text = f"💼 BUSINESS IMPACT\n\n"
        
        if cohort_scores and 'Total Users' in self.comparison_results:
            # Calculate weighted impact (performance * user base)
            impact_scores = {}
            for cohort in self.cohorts:
                if cohort in cohort_scores and cohort in self.comparison_results['Total Users']:
                    performance = cohort_scores[cohort]
                    user_base = self.comparison_results['Total Users'][cohort]['value']
                    # Weighted impact score
                    impact_scores[cohort] = (performance + 100) * user_base / 1000  # Normalized
            
            if impact_scores:
                top_impact = max(impact_scores, key=impact_scores.get)
                impact_text += f"🎯 HIGHEST BUSINESS IMPACT:\n{top_impact}\n"
                impact_text += f"Impact Score: {impact_scores[top_impact]:.1f}\n\n"
                
                impact_text += "📊 IMPACT RANKING:\n"
                sorted_impact = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)
                for i, (cohort, score) in enumerate(sorted_impact, 1):
                    impact_text += f"{i}. {cohort}: {score:.1f}\n"
        
        ax7.text(0.05, 0.95, impact_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # 8. Action Items
        ax8 = axes[2, 1]
        ax8.axis('off')
        
        action_text = f"🎯 ACTION ITEMS\n\n"
        
        # Generate specific action items based on analysis
        if cohort_scores:
            champion = max(cohort_scores, key=cohort_scores.get)
            weakest = min(cohort_scores, key=cohort_scores.get)
            
            action_text += f"1. SCALE SUCCESS:\n   Study {champion} practices\n   Apply to other cohorts\n\n"
            action_text += f"2. URGENT FIXES:\n   Address {weakest} gaps\n   Focus on biggest opportunities\n\n"
            
            # Find systematic issues
            if gap_analysis:
                biggest_gap = gap_analysis[0]
                if biggest_gap['gap'] > 30:
                    action_text += f"3. CRITICAL ISSUE:\n   {biggest_gap['metric'][:30]}...\n   {biggest_gap['gap']:.1f}% performance gap\n\n"
            
            action_text += f"4. MONITOR & OPTIMIZE:\n   Track key metrics weekly\n   A/B test improvements"
        
        ax8.text(0.05, 0.95, action_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        # 9. Executive Summary
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        exec_text = f"📋 EXECUTIVE SUMMARY\n\n"
        
        if cohort_scores:
            champion = max(cohort_scores, key=cohort_scores.get)
            weakest = min(cohort_scores, key=cohort_scores.get)
            
            exec_text += f"🏆 WINNER: {champion}\n"
            exec_text += f"📊 Score: {cohort_scores[champion]:+.1f}%\n\n"
            
            exec_text += f"⚠️ FOCUS AREA: {weakest}\n"
            exec_text += f"📊 Score: {cohort_scores[weakest]:+.1f}%\n\n"
            
            # Overall assessment
            avg_performance = np.mean(list(cohort_scores.values()))
            performance_spread = max(cohort_scores.values()) - min(cohort_scores.values())
            
            exec_text += f"📈 AVG PERFORMANCE: {avg_performance:+.1f}%\n"
            exec_text += f"📊 PERFORMANCE SPREAD: {performance_spread:.1f}pp\n\n"
            
            if performance_spread > 20:
                exec_text += "🚨 HIGH DISPARITY\nSignificant optimization needed"
            elif performance_spread > 10:
                exec_text += "⚠️ MODERATE GAPS\nTargeted improvements required"
            else:
                exec_text += "✅ BALANCED PERFORMANCE\nFine-tuning opportunities"
        
        ax9.text(0.05, 0.95, exec_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{self.folder_name}/05_comprehensive_summary.png', dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive summary dashboard saved")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print(f"📋 Generating {self.comparison_name} Comparison Report...")
        
        report = []
        report.append("="*80)
        report.append(f"📊 {self.comparison_name.upper()} COMPREHENSIVE COMPARISON REPORT")
        report.append("="*80)
        report.append("")
        
        # Executive Summary
        cohort_scores = {}
        for cohort in self.cohorts:
            scores = []
            for metric, data in self.comparison_results.items():
                if cohort in data and 'vs_benchmark' in data[cohort]:
                    scores.append(data[cohort]['vs_benchmark'])
            if scores:
                cohort_scores[cohort] = np.mean(scores)
        
        if cohort_scores:
            champion = max(cohort_scores, key=cohort_scores.get)
            weakest = min(cohort_scores, key=cohort_scores.get)
            
            report.append("🎯 EXECUTIVE SUMMARY")
            report.append("-" * 40)
            report.append(f"Champion: {champion} ({cohort_scores[champion]:+.1f}% avg performance)")
            report.append(f"Needs Focus: {weakest} ({cohort_scores[weakest]:+.1f}% avg performance)")
            report.append(f"Performance Gap: {cohort_scores[champion] - cohort_scores[weakest]:.1f} percentage points")
            report.append("")
        
        # Detailed Comparison
        report.append("📊 DETAILED COMPARISON ANALYSIS")
        report.append("-" * 40)
        
        # Count wins for each cohort
        cohort_wins = {cohort: 0 for cohort in self.cohorts}
        total_comparisons = 0
        
        for metric, data in self.comparison_results.items():
            if 'analysis' in data:
                winner = data['analysis']['winner']
                if winner in cohort_wins:
                    cohort_wins[winner] += 1
                total_comparisons += 1
        
        report.append("🏆 FEATURE WINS LEADERBOARD:")
        sorted_winners = sorted(cohort_wins.items(), key=lambda x: x[1], reverse=True)
        for i, (cohort, wins) in enumerate(sorted_winners, 1):
            win_percentage = (wins / total_comparisons * 100) if total_comparisons > 0 else 0
            report.append(f"   {i}. {cohort}: {wins}/{total_comparisons} wins ({win_percentage:.1f}%)")
        report.append("")
        
        # Top Performance Gaps
        gap_analysis = []
        for metric, data in self.comparison_results.items():
            if 'analysis' in data and data['analysis']['performance_gap'] > 5:
                gap_analysis.append({
                    'metric': metric,
                    'gap': data['analysis']['performance_gap'],
                    'winner': data['analysis']['winner'],
                    'loser': data['analysis']['worst_cohort']
                })
        
        gap_analysis.sort(key=lambda x: x['gap'], reverse=True)
        
        if gap_analysis:
            report.append("📊 TOP PERFORMANCE GAPS:")
            for i, gap in enumerate(gap_analysis[:5], 1):
                report.append(f"   {i}. {gap['metric']}")
                report.append(f"      {gap['winner']} outperforms {gap['loser']} by {gap['gap']:.1f}%")
            report.append("")
        
        # Strategic Recommendations
        report.append("💡 STRATEGIC RECOMMENDATIONS")
        report.append("-" * 40)
        
        if cohort_scores and gap_analysis:
            report.append(f"1. LEVERAGE SUCCESS PATTERNS:")
            report.append(f"   - Scale {champion} practices across organization")
            report.append(f"   - Identify and document {champion} success factors")
            report.append("")
            
            report.append(f"2. ADDRESS CRITICAL GAPS:")
            report.append(f"   - Priority focus on {weakest} performance issues")
            if gap_analysis[0]['gap'] > 20:
                report.append(f"   - Urgent attention to {gap_analysis[0]['metric']}")
            report.append("")
            
            report.append(f"3. OPTIMIZATION OPPORTUNITIES:")
            report.append(f"   - Implement cross-cohort learning programs")
            report.append(f"   - Establish performance monitoring dashboards")
            report.append(f"   - Regular comparative analysis reviews")
            report.append("")
        
        # Business Impact Assessment
        report.append("💼 BUSINESS IMPACT ASSESSMENT")
        report.append("-" * 40)
        
        if 'Total Users' in self.comparison_results:
            total_users_analyzed = 0
            for cohort in self.cohorts:
                if cohort in self.comparison_results['Total Users']:
                    users = self.comparison_results['Total Users'][cohort]['value']
                    total_users_analyzed += users
                    performance = cohort_scores.get(cohort, 0)
                    report.append(f"{cohort}: {users:,.0f} users ({performance:+.1f}% performance)")
            
            report.append(f"Total Users Analyzed: {total_users_analyzed:,.0f}")
            
            # Calculate weighted performance impact
            if cohort_scores:
                weighted_impact = 0
                for cohort in self.cohorts:
                    if cohort in self.comparison_results['Total Users'] and cohort in cohort_scores:
                        users = self.comparison_results['Total Users'][cohort]['value']
                        performance = cohort_scores[cohort]
                        weighted_impact += (performance * users) / total_users_analyzed
                
                report.append(f"Weighted Performance Impact: {weighted_impact:+.1f}%")
        
        report.append("")
        
        # Save report
        safe_filename = self.comparison_name.replace(' ', '_').replace('vs', '_vs_').lower()
        with open(f'{self.folder_name}/{safe_filename}_comparison_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print(f"📋 {self.comparison_name} comparison report saved")
        
        return cohort_scores

def generate_comparison_analysis(comparison_name: str, cohorts: List[str], folder_name: str):
    """Generate complete comparison analysis"""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING {comparison_name.upper()} COMPARISON ANALYSIS")
    print(f"{'='*60}")
    
    # Initialize analyzer
    analyzer = ComparativeAnalysis('Cohort Wise Analysis Fam 2.0 - Sheet1.csv', comparison_name, cohorts, folder_name)
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Extract cohorts data
    cohorts_data, benchmark_data = analyzer.extract_cohorts_data()
    
    # Run comparative analysis
    comparison_results = analyzer.analyze_comparative_performance()
    
    # Create all visualizations
    print(f"\n📊 Creating visualizations for {comparison_name}...")
    analyzer.create_overall_comparison_viz()          # PNG 1
    analyzer.create_feature_comparison_viz()          # PNG 2  
    analyzer.create_engagement_analysis_viz()         # PNG 3
    analyzer.create_performance_gaps_viz()            # PNG 4
    analyzer.create_comprehensive_summary_viz()       # PNG 5
    
    # Generate comparison report
    cohort_scores = analyzer.generate_comparison_report()
    
    print(f"\n✅ {comparison_name} Comparison Analysis Complete!")
    print(f"📁 Generated 5 PNG files + comparison report in '{folder_name}' folder")
    
    if cohort_scores:
        champion = max(cohort_scores, key=cohort_scores.get)
        weakest = min(cohort_scores, key=cohort_scores.get)
        print(f"🏆 Winner: {champion} ({cohort_scores[champion]:+.1f}%)")
        print(f"⚠️ Focus Area: {weakest} ({cohort_scores[weakest]:+.1f}%)")
    
    return cohort_scores

def main():
    """Main function to generate all comparison analyses"""
    print("🚀 STARTING ALL COMPARATIVE ANALYSES")
    print("="*80)
    
    # Define all comparisons
    comparisons = [
        {
            'name': '18+ vs 18-',
            'cohorts': ['18+', '18-'],
            'folder': '18+ vs 18- Analysis'
        },
        {
            'name': 'Android vs iOS',
            'cohorts': ['Android', 'IOS '],  # Note: CSV has space after IOS
            'folder': 'Android vs iOS Analysis'
        },
        {
            'name': 'SLS vs DM vs Bubble',
            'cohorts': ['SLS ', 'DM', 'Bubble'],  # Note: CSV has space after SLS
            'folder': 'SLS vs DM vs Bubble Analysis'
        },
        {
            'name': 'PPI vs TPAP vs Both',
            'cohorts': ['PPI', 'TPAP', 'Both'],
            'folder': 'PPI vs TPAP vs Both Analysis'
        },
        {
            'name': 'Overall vs Ultra Users',
            'cohorts': ['Combine All', 'Ultra Users'],
            'folder': 'Overall vs Ultra Users Analysis'
        }
    ]
    
    results = []
    
    for comparison in comparisons:
        try:
            result = generate_comparison_analysis(
                comparison['name'], 
                comparison['cohorts'], 
                comparison['folder']
            )
            if result:
                results.append({
                    'name': comparison['name'],
                    'scores': result
                })
        except Exception as e:
            print(f"❌ Error in {comparison['name']}: {str(e)}")
            continue
    
    # Generate final summary
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARATIVE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print("\n🏆 COMPARISON WINNERS:")
    for result in results:
        if result['scores']:
            winner = max(result['scores'], key=result['scores'].get)
            winner_score = result['scores'][winner]
            print(f"   {result['name']}: {winner} ({winner_score:+.1f}%)")
    
    print("\n✅ ALL COMPARATIVE ANALYSES COMPLETED!")
    print("📁 Each comparison folder contains:")
    print("   📊 01_overall_comparison.png")
    print("   🔍 02_feature_comparison.png")
    print("   💬 03_engagement_analysis.png")
    print("   📊 04_performance_gaps.png")
    print("   📊 05_comprehensive_summary.png")
    print("   📋 [comparison]_report.txt")

if __name__ == "__main__":
    main()
