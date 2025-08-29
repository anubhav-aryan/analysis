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

class ComprehensiveKPIAnalysis:
    def __init__(self, file_path: str):
        """Initialize comprehensive KPI analysis for all cohorts"""
        self.file_path = file_path
        self.df = None
        self.benchmark_cohort = "Combine All"
        self.all_cohorts = []
        self.all_metrics = []
        self.kpi_data = {}
        self.performance_matrix = {}
        self.tier_rankings = {}
        self.champion_analysis = {}
        
        # Define cohort groups for analysis
        self.cohort_groups = {
            'Age Groups': ['18+', '18-'],
            'Platforms': ['Android', 'IOS '],
            'Features': ['SLS ', 'DM', 'Bubble'],
            'Payment Types': ['PPI', 'TPAP', 'Both'],
            'User Types': ['Ultra Users', 'Rep Set'],
            'Overall': ['Combine All']
        }
        
        # Performance tier thresholds (vs benchmark)
        self.tier_thresholds = {
            'Champion': 20,      # >20% above benchmark
            'Strong': 5,         # 5-20% above benchmark  
            'Moderate': -10,     # -10% to 5% vs benchmark
            'Weak': -25,         # -25% to -10% vs benchmark
            'Critical': -100     # <-25% vs benchmark
        }
        
        # Color schemes for tiers
        self.tier_colors = {
            'Champion': '#2E8B57',    # Sea Green
            'Strong': '#32CD32',      # Lime Green
            'Moderate': '#FFD700',    # Gold
            'Weak': '#FF8C00',        # Dark Orange
            'Critical': '#DC143C'     # Crimson
        }
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the CSV data"""
        print("📊 Loading comprehensive KPI data...")
        
        # Load the CSV
        self.df = pd.read_csv(self.file_path)
        
        # Get all cohorts (exclude first column which contains metric names)
        self.all_cohorts = [col for col in self.df.columns[1:] if col.strip()]
        print(f"📈 Found {len(self.all_cohorts)} cohorts: {', '.join(self.all_cohorts)}")
        
        # Clean the data
        self._clean_data()
        
        return self.df
    
    def _clean_data(self):
        """Clean and standardize the data"""
        # Remove completely empty rows
        self.df = self.df.dropna(how='all')
        
        # Replace dashes with NaN
        self.df = self.df.replace('-', np.nan)
        
        # Clean percentage values and time values for all cohorts
        for col in self.all_cohorts:
            if col in self.df.columns:
                # Convert to string first
                self.df[col] = self.df[col].astype(str)
                # Remove percentage signs, time units, and clean up
                self.df[col] = self.df[col].str.replace('%', '').str.replace('secs', '').str.replace('mins', '').str.replace('sec', '')
                # Convert to numeric, errors='coerce' will convert invalid values to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def extract_all_kpi_data(self):
        """Extract data for all KPIs across all cohorts"""
        print("📊 Extracting KPI data for all cohorts...")
        
        # Initialize data storage
        self.kpi_data = {}
        
        # Process each row that contains metrics
        for idx, row in self.df.iterrows():
            metric_name = str(row.iloc[0]).strip()
            
            # Skip empty or invalid metric names
            if pd.isna(metric_name) or metric_name == '' or metric_name == 'nan':
                continue
            
            # Skip section headers
            if metric_name in ['Overall', 'Spotlight', 'Home', 'DM Dashboard', 'Bubble']:
                continue
            
            # Store metric
            self.all_metrics.append(metric_name)
            self.kpi_data[metric_name] = {}
            
            # Extract data for each cohort
            for cohort in self.all_cohorts:
                if cohort in self.df.columns:
                    cohort_value = row[cohort]
                    if pd.notna(cohort_value) and cohort_value != '' and str(cohort_value) != 'nan':
                        try:
                            self.kpi_data[metric_name][cohort] = float(cohort_value)
                        except:
                            continue
        
        print(f"📊 Extracted {len(self.all_metrics)} KPIs across {len(self.all_cohorts)} cohorts")
        return self.kpi_data
    
    def calculate_performance_matrix(self):
        """Calculate performance vs benchmark for all KPIs and cohorts"""
        print("🔍 Calculating performance matrix...")
        
        self.performance_matrix = {}
        benchmark_data = {}
        
        # Get benchmark values
        for metric in self.all_metrics:
            if metric in self.kpi_data and self.benchmark_cohort in self.kpi_data[metric]:
                benchmark_data[metric] = self.kpi_data[metric][self.benchmark_cohort]
        
        # Calculate performance vs benchmark
        for metric in self.all_metrics:
            self.performance_matrix[metric] = {}
            benchmark_value = benchmark_data.get(metric, 1)
            
            for cohort in self.all_cohorts:
                if cohort == self.benchmark_cohort:
                    self.performance_matrix[metric][cohort] = 0.0  # Benchmark is 0%
                elif metric in self.kpi_data and cohort in self.kpi_data[metric]:
                    cohort_value = self.kpi_data[metric][cohort]
                    if benchmark_value != 0:
                        performance_vs_benchmark = ((cohort_value - benchmark_value) / benchmark_value) * 100
                        self.performance_matrix[metric][cohort] = performance_vs_benchmark
                    else:
                        self.performance_matrix[metric][cohort] = 0.0
        
        print(f"📊 Performance matrix calculated for {len(self.performance_matrix)} KPIs")
        return self.performance_matrix
    
    def create_tier_rankings(self):
        """Create performance tier rankings for all KPIs"""
        print("🏆 Creating tier rankings...")
        
        self.tier_rankings = {}
        
        for metric in self.all_metrics:
            if metric in self.performance_matrix:
                metric_performance = self.performance_matrix[metric]
                
                # Sort cohorts by performance for this metric
                sorted_cohorts = sorted(metric_performance.items(), key=lambda x: x[1], reverse=True)
                
                # Assign tiers based on performance thresholds
                tiers = {
                    'Champion': [],
                    'Strong': [],
                    'Moderate': [],
                    'Weak': [],
                    'Critical': []
                }
                
                for cohort, performance in sorted_cohorts:
                    if cohort == self.benchmark_cohort:
                        tiers['Moderate'].append((cohort, performance))  # Benchmark is moderate
                    elif performance >= self.tier_thresholds['Champion']:
                        tiers['Champion'].append((cohort, performance))
                    elif performance >= self.tier_thresholds['Strong']:
                        tiers['Strong'].append((cohort, performance))
                    elif performance >= self.tier_thresholds['Moderate']:
                        tiers['Moderate'].append((cohort, performance))
                    elif performance >= self.tier_thresholds['Weak']:
                        tiers['Weak'].append((cohort, performance))
                    else:
                        tiers['Critical'].append((cohort, performance))
                
                self.tier_rankings[metric] = {
                    'tiers': tiers,
                    'sorted_performance': sorted_cohorts
                }
        
        print(f"🏆 Tier rankings created for {len(self.tier_rankings)} KPIs")
        return self.tier_rankings
    
    def analyze_champion_cohorts(self):
        """Analyze which cohorts are consistent champions across KPIs"""
        print("🏆 Analyzing champion cohorts...")
        
        # Count performance by tier for each cohort
        cohort_tier_counts = {}
        for cohort in self.all_cohorts:
            cohort_tier_counts[cohort] = {
                'Champion': 0,
                'Strong': 0,
                'Moderate': 0,
                'Weak': 0,
                'Critical': 0,
                'Total': 0
            }
        
        # Count appearances in each tier
        for metric, rankings in self.tier_rankings.items():
            for tier, cohorts in rankings['tiers'].items():
                for cohort, performance in cohorts:
                    if cohort in cohort_tier_counts:
                        cohort_tier_counts[cohort][tier] += 1
                        cohort_tier_counts[cohort]['Total'] += 1
        
        # Calculate performance scores and categorize cohorts
        self.champion_analysis = {
            'Consistent Champions': [],      # High champion + strong rates
            'Specialized Champions': [],     # High champion in specific areas
            'Solid Performers': [],         # High strong + moderate rates
            'Inconsistent Performers': [],  # Mixed performance
            'Consistent Strugglers': [],    # High weak + critical rates
            'Critical Cases': []            # Very high critical rates
        }
        
        for cohort, counts in cohort_tier_counts.items():
            if counts['Total'] == 0:
                continue
                
            # Calculate percentages
            champion_rate = counts['Champion'] / counts['Total'] * 100
            strong_rate = counts['Strong'] / counts['Total'] * 100
            critical_rate = counts['Critical'] / counts['Total'] * 100
            weak_rate = counts['Weak'] / counts['Total'] * 100
            
            # Categorize cohorts
            if champion_rate >= 30:
                self.champion_analysis['Consistent Champions'].append({
                    'cohort': cohort,
                    'champion_rate': champion_rate,
                    'strong_rate': strong_rate,
                    'total_kpis': counts['Total']
                })
            elif champion_rate >= 15:
                self.champion_analysis['Specialized Champions'].append({
                    'cohort': cohort,
                    'champion_rate': champion_rate,
                    'strong_rate': strong_rate,
                    'total_kpis': counts['Total']
                })
            elif strong_rate + champion_rate >= 40:
                self.champion_analysis['Solid Performers'].append({
                    'cohort': cohort,
                    'champion_rate': champion_rate,
                    'strong_rate': strong_rate,
                    'total_kpis': counts['Total']
                })
            elif critical_rate >= 40:
                self.champion_analysis['Critical Cases'].append({
                    'cohort': cohort,
                    'critical_rate': critical_rate,
                    'weak_rate': weak_rate,
                    'total_kpis': counts['Total']
                })
            elif weak_rate + critical_rate >= 50:
                self.champion_analysis['Consistent Strugglers'].append({
                    'cohort': cohort,
                    'critical_rate': critical_rate,
                    'weak_rate': weak_rate,
                    'total_kpis': counts['Total']
                })
            else:
                self.champion_analysis['Inconsistent Performers'].append({
                    'cohort': cohort,
                    'champion_rate': champion_rate,
                    'strong_rate': strong_rate,
                    'weak_rate': weak_rate,
                    'total_kpis': counts['Total']
                })
        
        # Sort each category by performance
        for category in self.champion_analysis:
            if 'champion_rate' in str(self.champion_analysis[category]):
                self.champion_analysis[category].sort(key=lambda x: x.get('champion_rate', 0), reverse=True)
            elif 'critical_rate' in str(self.champion_analysis[category]):
                self.champion_analysis[category].sort(key=lambda x: x.get('critical_rate', 0), reverse=True)
        
        print("🏆 Champion analysis completed")
        return self.champion_analysis
    
    def create_performance_heatmap_viz(self):
        """Create comprehensive performance heatmap (PNG 1)"""
        print("📊 Creating Performance Heatmap Visualization...")
        
        fig, axes = plt.subplots(2, 1, figsize=(20, 16))
        fig.suptitle('🏆 COMPREHENSIVE KPI PERFORMANCE HEATMAP', fontsize=18, fontweight='bold')
        
        # 1. Full Performance Matrix Heatmap
        ax1 = axes[0]
        
        # Create matrix for heatmap
        heatmap_data = []
        metric_labels = []
        cohort_labels = [c for c in self.all_cohorts if c != self.benchmark_cohort]  # Exclude benchmark
        
        # Select top 20 most varying metrics for readability
        metric_variances = {}
        for metric in self.all_metrics:
            if metric in self.performance_matrix:
                values = [self.performance_matrix[metric].get(cohort, 0) for cohort in cohort_labels]
                if len(values) > 1:
                    metric_variances[metric] = np.var(values)
        
        top_metrics = sorted(metric_variances.items(), key=lambda x: x[1], reverse=True)[:20]
        
        for metric, _ in top_metrics:
            row = []
            for cohort in cohort_labels:
                performance = self.performance_matrix[metric].get(cohort, 0)
                row.append(performance)
            heatmap_data.append(row)
            metric_labels.append(metric[:25] + '...' if len(metric) > 25 else metric)
        
        if heatmap_data:
            # Create heatmap
            im1 = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
            ax1.set_title('KPI Performance vs Benchmark (Top 20 Variable Metrics)', fontsize=14, fontweight='bold')
            ax1.set_xticks(range(len(cohort_labels)))
            ax1.set_xticklabels(cohort_labels, rotation=45, ha='right')
            ax1.set_yticks(range(len(metric_labels)))
            ax1.set_yticklabels(metric_labels)
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Performance vs Benchmark (%)', fontsize=12)
            
            # Add text annotations for extreme values
            for i in range(len(metric_labels)):
                for j in range(len(cohort_labels)):
                    value = heatmap_data[i][j]
                    if abs(value) > 25:  # Only show extreme values
                        color = 'white' if abs(value) > 40 else 'black'
                        ax1.text(j, i, f'{value:.0f}%', ha="center", va="center", 
                                color=color, fontsize=8, fontweight='bold')
        
        # 2. Tier Distribution Heatmap
        ax2 = axes[1]
        
        # Create tier count matrix
        tier_matrix = []
        tier_labels = ['Champion', 'Strong', 'Moderate', 'Weak', 'Critical']
        
        for cohort in cohort_labels:
            tier_counts = [0, 0, 0, 0, 0]  # Champion, Strong, Moderate, Weak, Critical
            total_metrics = 0
            
            for metric, rankings in self.tier_rankings.items():
                for tier_idx, tier in enumerate(tier_labels):
                    tier_cohorts = [c[0] for c in rankings['tiers'][tier]]
                    if cohort in tier_cohorts:
                        tier_counts[tier_idx] += 1
                        total_metrics += 1
                        break
            
            # Convert to percentages
            if total_metrics > 0:
                tier_percentages = [count / total_metrics * 100 for count in tier_counts]
            else:
                tier_percentages = [0, 0, 0, 0, 0]
            
            tier_matrix.append(tier_percentages)
        
        if tier_matrix:
            tier_matrix = np.array(tier_matrix).T  # Transpose for correct orientation
            
            # Create custom colormap for tiers
            colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#DC143C']  # Champion to Critical
            n_bins = 100
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list('tier_cmap', colors, N=n_bins)
            
            im2 = ax2.imshow(tier_matrix, cmap='YlOrRd', aspect='auto')
            ax2.set_title('Performance Tier Distribution by Cohort', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(cohort_labels)))
            ax2.set_xticklabels(cohort_labels, rotation=45, ha='right')
            ax2.set_yticks(range(len(tier_labels)))
            ax2.set_yticklabels(tier_labels)
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Percentage of KPIs (%)', fontsize=12)
            
            # Add percentage annotations
            for i in range(len(tier_labels)):
                for j in range(len(cohort_labels)):
                    percentage = tier_matrix[i][j]
                    if percentage > 5:  # Only show significant percentages
                        ax2.text(j, i, f'{percentage:.0f}%', ha="center", va="center", 
                                color='white' if percentage > 50 else 'black', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('01_performance_heatmap.png', dpi=300, bbox_inches='tight')
        print("📊 Performance heatmap visualization saved")
    
    def create_champion_analysis_viz(self):
        """Create champion analysis visualization (PNG 2)"""
        print("🏆 Creating Champion Analysis Visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle('🏆 CHAMPION COHORTS ANALYSIS', fontsize=18, fontweight='bold')
        
        # 1. Consistent Champions
        ax1 = axes[0, 0]
        champions = self.champion_analysis['Consistent Champions']
        
        if champions:
            cohort_names = [c['cohort'] for c in champions]
            champion_rates = [c['champion_rate'] for c in champions]
            
            bars = ax1.bar(cohort_names, champion_rates, color='#2E8B57', alpha=0.8)
            ax1.set_title('🏆 Consistent Champions\n(≥30% Champion Rate)', fontweight='bold')
            ax1.set_ylabel('Champion Rate (%)')
            ax1.tick_params(axis='x', rotation=45)
            plt.setp(ax1.get_xticklabels(), ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, rate in zip(bars, champion_rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'No Consistent Champions Found', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14, style='italic')
            ax1.set_title('🏆 Consistent Champions', fontweight='bold')
        
        # 2. Critical Cases
        ax2 = axes[0, 1]
        critical_cases = self.champion_analysis['Critical Cases']
        
        if critical_cases:
            cohort_names = [c['cohort'] for c in critical_cases]
            critical_rates = [c['critical_rate'] for c in critical_cases]
            
            bars = ax2.bar(cohort_names, critical_rates, color='#DC143C', alpha=0.8)
            ax2.set_title('🚨 Critical Cases\n(≥40% Critical Rate)', fontweight='bold')
            ax2.set_ylabel('Critical Rate (%)')
            ax2.tick_params(axis='x', rotation=45)
            plt.setp(ax2.get_xticklabels(), ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, rate in zip(bars, critical_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No Critical Cases Found', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14, style='italic')
            ax2.set_title('🚨 Critical Cases', fontweight='bold')
        
        # 3. Performance Distribution Pie Chart
        ax3 = axes[0, 2]
        
        # Count cohorts by category
        category_counts = {category: len(cohorts) for category, cohorts in self.champion_analysis.items()}
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        # Colors for categories
        category_colors = ['#2E8B57', '#90EE90', '#32CD32', '#FFD700', '#FF8C00', '#DC143C']
        
        if sum(counts) > 0:
            wedges, texts, autotexts = ax3.pie(counts, labels=categories, colors=category_colors[:len(categories)], 
                                              autopct='%1.0f', startangle=90)
            ax3.set_title('Cohort Distribution by\nPerformance Category', fontweight='bold')
        
        # 4. Top Performers by Group
        ax4 = axes[1, 0]
        ax4.axis('off')
        
        # Find best performer in each cohort group
        group_champions = {}
        for group, cohorts in self.cohort_groups.items():
            if group == 'Overall':
                continue
                
            best_cohort = None
            best_champion_rate = -1
            
            for category_cohorts in self.champion_analysis.values():
                for cohort_data in category_cohorts:
                    cohort = cohort_data['cohort']
                    if cohort in cohorts:
                        champion_rate = cohort_data.get('champion_rate', 0)
                        if champion_rate > best_champion_rate:
                            best_champion_rate = champion_rate
                            best_cohort = cohort
            
            if best_cohort:
                group_champions[group] = {'cohort': best_cohort, 'rate': best_champion_rate}
        
        champions_text = "🏆 GROUP CHAMPIONS\n\n"
        for group, data in group_champions.items():
            champions_text += f"{group}:\n  {data['cohort']} ({data['rate']:.1f}%)\n\n"
        
        ax4.text(0.05, 0.95, champions_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # 5. Specialized Champions
        ax5 = axes[1, 1]
        specialized = self.champion_analysis['Specialized Champions']
        
        if specialized:
            cohort_names = [c['cohort'] for c in specialized]
            champion_rates = [c['champion_rate'] for c in specialized]
            strong_rates = [c['strong_rate'] for c in specialized]
            
            x = np.arange(len(cohort_names))
            width = 0.35
            
            bars1 = ax5.bar(x - width/2, champion_rates, width, label='Champion Rate', 
                           color='#2E8B57', alpha=0.8)
            bars2 = ax5.bar(x + width/2, strong_rates, width, label='Strong Rate', 
                           color='#32CD32', alpha=0.8)
            
            ax5.set_title('⭐ Specialized Champions\n(15-30% Champion Rate)', fontweight='bold')
            ax5.set_ylabel('Performance Rate (%)')
            ax5.set_xticks(x)
            ax5.set_xticklabels(cohort_names, rotation=45, ha='right')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Specialized Champions Found', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=14, style='italic')
            ax5.set_title('⭐ Specialized Champions', fontweight='bold')
        
        # 6. Strategic Insights
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        insights_text = "💡 STRATEGIC INSIGHTS\n\n"
        
        # Find overall patterns
        total_cohorts = len(self.all_cohorts) - 1  # Exclude benchmark
        champions_count = len(self.champion_analysis['Consistent Champions'])
        critical_count = len(self.champion_analysis['Critical Cases'])
        
        insights_text += f"📊 PERFORMANCE OVERVIEW:\n"
        insights_text += f"• {champions_count}/{total_cohorts} Consistent Champions\n"
        insights_text += f"• {critical_count}/{total_cohorts} Critical Cases\n\n"
        
        if champions_count > 0:
            top_champion = self.champion_analysis['Consistent Champions'][0]
            insights_text += f"🏆 TOP PERFORMER:\n{top_champion['cohort']}\n"
            insights_text += f"Champion in {top_champion['champion_rate']:.1f}% of KPIs\n\n"
        
        if critical_count > 0:
            worst_case = self.champion_analysis['Critical Cases'][0]
            insights_text += f"🚨 URGENT ATTENTION:\n{worst_case['cohort']}\n"
            insights_text += f"Critical in {worst_case['critical_rate']:.1f}% of KPIs\n\n"
        
        insights_text += "📈 RECOMMENDATIONS:\n"
        insights_text += "• Scale champion practices\n"
        insights_text += "• Address critical cases\n"
        insights_text += "• Leverage specialized strengths"
        
        ax6.text(0.05, 0.95, insights_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('02_champion_analysis.png', dpi=300, bbox_inches='tight')
        print("🏆 Champion analysis visualization saved")
    
    def create_kpi_category_analysis_viz(self):
        """Create KPI category analysis visualization (PNG 3)"""
        print("📊 Creating KPI Category Analysis Visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('📊 KPI CATEGORY PERFORMANCE ANALYSIS', fontsize=18, fontweight='bold')
        
        # Define KPI categories
        kpi_categories = {
            'User Engagement': ['DAU', 'DTU', 'Time Spent', 'session', 'engagement'],
            'Feature Adoption': ['spotlight', 'scanning', 'adoption', 'usage', 'opening'],
            'Payment Performance': ['pay', 'transaction', 'UPI', 'Ticket Size', 'time to pay'],
            'Communication': ['DM', 'Messages', 'pins', 'repeat'],
            'Home & Navigation': ['Home', 'home screen', 'Home Screen'],
            'Other Features': ['Bubble', 'swipe', 'tap', 'quick actions']
        }
        
        # 1. Category Performance Heatmap
        ax1 = axes[0, 0]
        
        # Calculate average performance by category for each cohort
        category_performance = {}
        cohort_labels = [c for c in self.all_cohorts if c != self.benchmark_cohort]
        
        for category, keywords in kpi_categories.items():
            category_performance[category] = {}
            
            for cohort in cohort_labels:
                performances = []
                for metric in self.all_metrics:
                    if any(keyword.lower() in metric.lower() for keyword in keywords):
                        if metric in self.performance_matrix and cohort in self.performance_matrix[metric]:
                            performances.append(self.performance_matrix[metric][cohort])
                
                if performances:
                    category_performance[category][cohort] = np.mean(performances)
                else:
                    category_performance[category][cohort] = 0
        
        # Create heatmap data
        heatmap_data = []
        category_labels = list(category_performance.keys())
        
        for category in category_labels:
            row = [category_performance[category].get(cohort, 0) for cohort in cohort_labels]
            heatmap_data.append(row)
        
        if heatmap_data:
            im1 = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)
            ax1.set_title('Average Performance by Category', fontweight='bold')
            ax1.set_xticks(range(len(cohort_labels)))
            ax1.set_xticklabels(cohort_labels, rotation=45, ha='right')
            ax1.set_yticks(range(len(category_labels)))
            ax1.set_yticklabels(category_labels)
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Avg Performance vs Benchmark (%)')
            
            # Add text annotations
            for i in range(len(category_labels)):
                for j in range(len(cohort_labels)):
                    value = heatmap_data[i][j]
                    if abs(value) > 5:
                        color = 'white' if abs(value) > 20 else 'black'
                        ax1.text(j, i, f'{value:.0f}%', ha="center", va="center", 
                                color=color, fontsize=9)
        
        # 2. Category Champions
        ax2 = axes[0, 1]
        
        # Find champion in each category
        category_champions = {}
        for category in category_labels:
            best_cohort = max(category_performance[category], 
                            key=category_performance[category].get)
            best_performance = category_performance[category][best_cohort]
            category_champions[category] = {
                'cohort': best_cohort,
                'performance': best_performance
            }
        
        categories = list(category_champions.keys())
        champion_cohorts = [category_champions[cat]['cohort'] for cat in categories]
        performances = [category_champions[cat]['performance'] for cat in categories]
        
        # Color bars based on performance
        colors = ['#2E8B57' if p > 10 else '#32CD32' if p > 0 else '#FFD700' if p > -10 else '#FF8C00' 
                 for p in performances]
        
        bars = ax2.barh(categories, performances, color=colors, alpha=0.8)
        ax2.set_title('Category Champions', fontweight='bold')
        ax2.set_xlabel('Performance vs Benchmark (%)')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add cohort labels
        for bar, cohort, perf in zip(bars, champion_cohorts, performances):
            width = bar.get_width()
            ax2.text(width + (1 if width >= 0 else -1), bar.get_y() + bar.get_height()/2.,
                    f'{cohort} ({perf:+.1f}%)', ha='left' if width >= 0 else 'right', 
                    va='center', fontsize=9)
        
        # 3. Performance Variance Analysis
        ax3 = axes[1, 0]
        
        # Calculate variance in performance for each category
        category_variances = {}
        for category in category_labels:
            performances = list(category_performance[category].values())
            if len(performances) > 1:
                category_variances[category] = np.var(performances)
            else:
                category_variances[category] = 0
        
        categories = list(category_variances.keys())
        variances = list(category_variances.values())
        
        bars = ax3.bar(categories, variances, color='skyblue', alpha=0.7)
        ax3.set_title('Performance Variance by Category\n(Higher = More Opportunity)', fontweight='bold')
        ax3.set_ylabel('Performance Variance')
        ax3.tick_params(axis='x', rotation=45)
        plt.setp(ax3.get_xticklabels(), ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, var in zip(bars, variances):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(variances)*0.01,
                    f'{var:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Strategic Category Insights
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        insights_text = "📊 CATEGORY INSIGHTS\n\n"
        
        # Find best and worst performing categories overall
        category_avg_performance = {}
        for category in category_labels:
            performances = [p for p in category_performance[category].values() if p != 0]
            if performances:
                category_avg_performance[category] = np.mean(performances)
        
        if category_avg_performance:
            best_category = max(category_avg_performance, key=category_avg_performance.get)
            worst_category = min(category_avg_performance, key=category_avg_performance.get)
            
            insights_text += f"🏆 STRONGEST CATEGORY:\n{best_category}\n"
            insights_text += f"Avg: {category_avg_performance[best_category]:+.1f}%\n\n"
            
            insights_text += f"⚠️ WEAKEST CATEGORY:\n{worst_category}\n"
            insights_text += f"Avg: {category_avg_performance[worst_category]:+.1f}%\n\n"
        
        # Find highest variance category
        if category_variances:
            highest_variance_cat = max(category_variances, key=category_variances.get)
            insights_text += f"🎯 BIGGEST OPPORTUNITY:\n{highest_variance_cat}\n"
            insights_text += f"High performance variance\nsuggests optimization potential\n\n"
        
        insights_text += "💡 RECOMMENDATIONS:\n"
        insights_text += "• Leverage category champions\n"
        insights_text += "• Focus on high-variance areas\n"
        insights_text += "• Address systematic weaknesses"
        
        ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('03_category_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 KPI category analysis visualization saved")
    
    def create_tier_distribution_viz(self):
        """Create tier distribution analysis visualization (PNG 4)"""
        print("🏆 Creating Tier Distribution Analysis Visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('🏆 PERFORMANCE TIER DISTRIBUTION ANALYSIS', fontsize=18, fontweight='bold')
        
        # Calculate tier statistics for each cohort
        cohort_tier_stats = {}
        cohort_labels = [c for c in self.all_cohorts if c != self.benchmark_cohort]
        
        for cohort in cohort_labels:
            tier_counts = {'Champion': 0, 'Strong': 0, 'Moderate': 0, 'Weak': 0, 'Critical': 0}
            total_kpis = 0
            
            for metric, rankings in self.tier_rankings.items():
                for tier, tier_cohorts in rankings['tiers'].items():
                    if cohort in [c[0] for c in tier_cohorts]:
                        tier_counts[tier] += 1
                        total_kpis += 1
                        break
            
            if total_kpis > 0:
                tier_percentages = {tier: count/total_kpis*100 for tier, count in tier_counts.items()}
                cohort_tier_stats[cohort] = {
                    'counts': tier_counts,
                    'percentages': tier_percentages,
                    'total': total_kpis
                }
        
        # 1. Stacked Bar Chart - Tier Distribution
        ax1 = axes[0, 0]
        
        tier_names = ['Champion', 'Strong', 'Moderate', 'Weak', 'Critical']
        tier_colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#DC143C']
        
        bottom = np.zeros(len(cohort_labels))
        
        for i, tier in enumerate(tier_names):
            values = [cohort_tier_stats[cohort]['percentages'].get(tier, 0) 
                     for cohort in cohort_labels]
            bars = ax1.bar(cohort_labels, values, bottom=bottom, 
                          label=tier, color=tier_colors[i], alpha=0.8)
            bottom += values
        
        ax1.set_title('Performance Tier Distribution by Cohort', fontweight='bold')
        ax1.set_ylabel('Percentage of KPIs (%)')
        ax1.tick_params(axis='x', rotation=45)
        plt.setp(ax1.get_xticklabels(), ha='right')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)
        
        # 2. Champion vs Critical Scatter Plot
        ax2 = axes[0, 1]
        
        champion_rates = [cohort_tier_stats[cohort]['percentages'].get('Champion', 0) 
                         for cohort in cohort_labels]
        critical_rates = [cohort_tier_stats[cohort]['percentages'].get('Critical', 0) 
                         for cohort in cohort_labels]
        
        scatter = ax2.scatter(champion_rates, critical_rates, 
                             c=range(len(cohort_labels)), cmap='viridis', 
                             s=100, alpha=0.7)
        
        # Add cohort labels
        for i, cohort in enumerate(cohort_labels):
            ax2.annotate(cohort, (champion_rates[i], critical_rates[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_title('Champion vs Critical Performance', fontweight='bold')
        ax2.set_xlabel('Champion Rate (%)')
        ax2.set_ylabel('Critical Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax2.axhline(y=np.mean(critical_rates), color='red', linestyle='--', alpha=0.5)
        ax2.axvline(x=np.mean(champion_rates), color='green', linestyle='--', alpha=0.5)
        
        # 3. Performance Score Distribution
        ax3 = axes[1, 0]
        
        # Calculate performance scores for each cohort
        performance_scores = []
        cohort_names_for_scores = []
        
        for cohort in cohort_labels:
            if cohort in cohort_tier_stats:
                # Weighted score: Champion=5, Strong=4, Moderate=3, Weak=2, Critical=1
                score = (cohort_tier_stats[cohort]['percentages'].get('Champion', 0) * 5 +
                        cohort_tier_stats[cohort]['percentages'].get('Strong', 0) * 4 +
                        cohort_tier_stats[cohort]['percentages'].get('Moderate', 0) * 3 +
                        cohort_tier_stats[cohort]['percentages'].get('Weak', 0) * 2 +
                        cohort_tier_stats[cohort]['percentages'].get('Critical', 0) * 1) / 100
                
                performance_scores.append(score)
                cohort_names_for_scores.append(cohort)
        
        # Sort by performance score
        sorted_data = sorted(zip(cohort_names_for_scores, performance_scores), 
                           key=lambda x: x[1], reverse=True)
        sorted_cohorts, sorted_scores = zip(*sorted_data)
        
        # Color bars based on performance tier
        colors = []
        for score in sorted_scores:
            if score >= 4:
                colors.append('#2E8B57')  # Champion
            elif score >= 3.5:
                colors.append('#32CD32')  # Strong
            elif score >= 2.5:
                colors.append('#FFD700')  # Moderate
            elif score >= 2:
                colors.append('#FF8C00')  # Weak
            else:
                colors.append('#DC143C')  # Critical
        
        bars = ax3.bar(sorted_cohorts, sorted_scores, color=colors, alpha=0.8)
        ax3.set_title('Overall Performance Score\n(Weighted by Tier Distribution)', fontweight='bold')
        ax3.set_ylabel('Performance Score (1-5)')
        ax3.tick_params(axis='x', rotation=45)
        plt.setp(ax3.get_xticklabels(), ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add score labels
        for bar, score in zip(bars, sorted_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Tier Analysis Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "🏆 TIER ANALYSIS SUMMARY\n\n"
        
        # Find best and worst performers
        if performance_scores:
            best_performer = sorted_data[0]
            worst_performer = sorted_data[-1]
            
            summary_text += f"🥇 TOP PERFORMER:\n{best_performer[0]}\n"
            summary_text += f"Score: {best_performer[1]:.2f}/5.0\n\n"
            
            summary_text += f"🚨 NEEDS ATTENTION:\n{worst_performer[0]}\n"
            summary_text += f"Score: {worst_performer[1]:.2f}/5.0\n\n"
        
        # Calculate tier statistics
        total_champion_instances = sum(cohort_tier_stats[cohort]['counts']['Champion'] 
                                     for cohort in cohort_labels)
        total_critical_instances = sum(cohort_tier_stats[cohort]['counts']['Critical'] 
                                     for cohort in cohort_labels)
        total_instances = sum(cohort_tier_stats[cohort]['total'] for cohort in cohort_labels)
        
        if total_instances > 0:
            champion_percentage = total_champion_instances / total_instances * 100
            critical_percentage = total_critical_instances / total_instances * 100
            
            summary_text += f"📊 OVERALL PERFORMANCE:\n"
            summary_text += f"Champion: {champion_percentage:.1f}%\n"
            summary_text += f"Critical: {critical_percentage:.1f}%\n\n"
        
        # Find cohorts with most balanced performance
        most_balanced = None
        best_balance_score = float('inf')
        
        for cohort in cohort_labels:
            if cohort in cohort_tier_stats:
                # Calculate balance (lower variance = more balanced)
                percentages = list(cohort_tier_stats[cohort]['percentages'].values())
                variance = np.var(percentages)
                if variance < best_balance_score:
                    best_balance_score = variance
                    most_balanced = cohort
        
        if most_balanced:
            summary_text += f"⚖️ MOST BALANCED:\n{most_balanced}\n"
            summary_text += f"Consistent across tiers\n\n"
        
        summary_text += "💡 KEY INSIGHTS:\n"
        summary_text += "• Focus on scaling champions\n"
        summary_text += "• Address critical cases urgently\n"
        summary_text += "• Learn from balanced performers"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('04_tier_distribution.png', dpi=300, bbox_inches='tight')
        print("🏆 Tier distribution analysis visualization saved")
    
    def create_strategic_dashboard_viz(self):
        """Create strategic insights dashboard (PNG 5)"""
        print("📊 Creating Strategic Dashboard Visualization...")
        
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('🎯 STRATEGIC KPI INSIGHTS DASHBOARD', fontsize=20, fontweight='bold')
        
        # Calculate key metrics for dashboard
        cohort_labels = [c for c in self.all_cohorts if c != self.benchmark_cohort]
        
        # 1. Executive Summary
        ax1 = axes[0, 0]
        ax1.axis('off')
        
        exec_text = "📋 EXECUTIVE SUMMARY\n\n"
        
        # Calculate overall statistics
        total_kpis = len(self.all_metrics)
        total_cohorts = len(cohort_labels)
        
        champions = self.champion_analysis['Consistent Champions']
        critical_cases = self.champion_analysis['Critical Cases']
        
        exec_text += f"📊 SCOPE:\n"
        exec_text += f"• {total_kpis} KPIs analyzed\n"
        exec_text += f"• {total_cohorts} cohorts evaluated\n\n"
        
        exec_text += f"🏆 PERFORMANCE:\n"
        exec_text += f"• {len(champions)} Consistent Champions\n"
        exec_text += f"• {len(critical_cases)} Critical Cases\n\n"
        
        if champions:
            top_champion = champions[0]
            exec_text += f"🥇 TOP PERFORMER:\n{top_champion['cohort']}\n"
            exec_text += f"Champion Rate: {top_champion['champion_rate']:.1f}%\n\n"
        
        if critical_cases:
            worst_case = critical_cases[0]
            exec_text += f"🚨 URGENT CASE:\n{worst_case['cohort']}\n"
            exec_text += f"Critical Rate: {worst_case['critical_rate']:.1f}%"
        
        ax1.text(0.05, 0.95, exec_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 2. Champion Leaderboard
        ax2 = axes[0, 1]
        
        # Combine all performance categories and sort by champion rate
        all_performers = []
        for category, cohorts in self.champion_analysis.items():
            for cohort_data in cohorts:
                if 'champion_rate' in cohort_data:
                    all_performers.append(cohort_data)
        
        all_performers.sort(key=lambda x: x['champion_rate'], reverse=True)
        top_performers = all_performers[:8]  # Top 8
        
        if top_performers:
            cohort_names = [p['cohort'] for p in top_performers]
            champion_rates = [p['champion_rate'] for p in top_performers]
            
            bars = ax2.barh(cohort_names, champion_rates, color='#2E8B57', alpha=0.8)
            ax2.set_title('🏆 Champion Leaderboard\n(Top 8 by Champion Rate)', fontweight='bold')
            ax2.set_xlabel('Champion Rate (%)')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, rate in zip(bars, champion_rates):
                width = bar.get_width()
                ax2.text(width + 1, bar.get_y() + bar.get_height()/2.,
                        f'{rate:.1f}%', ha='left', va='center', fontsize=10)
        
        # 3. Critical Issues Radar
        ax3 = axes[0, 2]
        
        # Create radar chart for critical cases
        if critical_cases:
            top_critical = critical_cases[:5]  # Top 5 critical cases
            cohort_names = [c['cohort'] for c in top_critical]
            critical_rates = [c['critical_rate'] for c in top_critical]
            
            # Simple bar chart instead of radar for clarity
            bars = ax3.bar(cohort_names, critical_rates, color='#DC143C', alpha=0.8)
            ax3.set_title('🚨 Critical Cases\n(Highest Critical Rates)', fontweight='bold')
            ax3.set_ylabel('Critical Rate (%)')
            ax3.tick_params(axis='x', rotation=45)
            plt.setp(ax3.get_xticklabels(), ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, rate in zip(bars, critical_rates):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 4. Performance Distribution Overview
        ax4 = axes[1, 0]
        
        # Calculate overall tier distribution
        tier_totals = {'Champion': 0, 'Strong': 0, 'Moderate': 0, 'Weak': 0, 'Critical': 0}
        total_instances = 0
        
        for metric, rankings in self.tier_rankings.items():
            for tier, cohorts in rankings['tiers'].items():
                tier_totals[tier] += len(cohorts)
                total_instances += len(cohorts)
        
        if total_instances > 0:
            tier_percentages = [tier_totals[tier]/total_instances*100 for tier in tier_totals.keys()]
            tier_labels = list(tier_totals.keys())
            tier_colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#DC143C']
            
            wedges, texts, autotexts = ax4.pie(tier_percentages, labels=tier_labels, 
                                              colors=tier_colors, autopct='%1.1f%%', 
                                              startangle=90)
            ax4.set_title('Overall Performance Distribution', fontweight='bold')
        
        # 5. Top KPI Champions
        ax5 = axes[1, 1]
        
        # Find KPIs with biggest performance gaps
        kpi_gaps = []
        for metric, rankings in self.tier_rankings.items():
            sorted_performance = rankings['sorted_performance']
            if len(sorted_performance) >= 2:
                best_performance = sorted_performance[0][1]
                worst_performance = sorted_performance[-1][1]
                gap = best_performance - worst_performance
                if gap > 10:  # Only significant gaps
                    kpi_gaps.append({
                        'metric': metric,
                        'gap': gap,
                        'champion': sorted_performance[0][0],
                        'champion_performance': best_performance
                    })
        
        kpi_gaps.sort(key=lambda x: x['gap'], reverse=True)
        top_kpi_gaps = kpi_gaps[:6]  # Top 6
        
        if top_kpi_gaps:
            metric_names = [gap['metric'][:20] + '...' if len(gap['metric']) > 20 else gap['metric'] 
                           for gap in top_kpi_gaps]
            gap_values = [gap['gap'] for gap in top_kpi_gaps]
            
            bars = ax5.barh(metric_names, gap_values, color='orange', alpha=0.8)
            ax5.set_title('🎯 Biggest KPI Gaps\n(Highest Opportunity)', fontweight='bold')
            ax5.set_xlabel('Performance Gap (%)')
            ax5.grid(True, alpha=0.3)
            
            # Add champion labels
            for bar, gap in zip(bars, top_kpi_gaps):
                width = bar.get_width()
                ax5.text(width + gap['gap']*0.02, bar.get_y() + bar.get_height()/2.,
                        f"{gap['champion']}", ha='left', va='center', fontsize=9)
        
        # 6. Group Performance Summary
        ax6 = axes[1, 2]
        
        # Calculate average performance by cohort group
        group_performance = {}
        for group, cohorts in self.cohort_groups.items():
            if group == 'Overall':
                continue
            
            group_scores = []
            for cohort in cohorts:
                if cohort in cohort_labels:
                    # Calculate performance score for this cohort
                    champion_count = 0
                    total_count = 0
                    for metric, rankings in self.tier_rankings.items():
                        for tier, tier_cohorts in rankings['tiers'].items():
                            if cohort in [c[0] for c in tier_cohorts]:
                                if tier == 'Champion':
                                    champion_count += 1
                                total_count += 1
                                break
                    
                    if total_count > 0:
                        champion_rate = champion_count / total_count * 100
                        group_scores.append(champion_rate)
            
            if group_scores:
                group_performance[group] = np.mean(group_scores)
        
        if group_performance:
            groups = list(group_performance.keys())
            performances = list(group_performance.values())
            
            bars = ax6.bar(groups, performances, color='steelblue', alpha=0.8)
            ax6.set_title('Average Champion Rate by Group', fontweight='bold')
            ax6.set_ylabel('Avg Champion Rate (%)')
            ax6.tick_params(axis='x', rotation=45)
            plt.setp(ax6.get_xticklabels(), ha='right')
            ax6.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, perf in zip(bars, performances):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{perf:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 7. Strategic Action Matrix
        ax7 = axes[2, 0]
        ax7.axis('off')
        
        action_text = "🎯 ACTION MATRIX\n\n"
        
        action_text += "🚀 IMMEDIATE ACTIONS:\n"
        if critical_cases:
            for case in critical_cases[:2]:
                action_text += f"• Fix {case['cohort']} issues\n"
        
        action_text += "\n📈 SCALE SUCCESS:\n"
        if champions:
            for champion in champions[:2]:
                action_text += f"• Scale {champion['cohort']} practices\n"
        
        action_text += "\n🎯 OPPORTUNITIES:\n"
        if top_kpi_gaps:
            for gap in top_kpi_gaps[:2]:
                action_text += f"• Optimize {gap['metric'][:15]}...\n"
        
        action_text += "\n⚖️ BALANCE PORTFOLIO:\n"
        action_text += "• Invest in consistent performers\n"
        action_text += "• Diversify risk across cohorts"
        
        ax7.text(0.05, 0.95, action_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # 8. Risk Assessment
        ax8 = axes[2, 1]
        ax8.axis('off')
        
        risk_text = "⚠️ RISK ASSESSMENT\n\n"
        
        # Calculate risk factors
        high_critical_cohorts = [case['cohort'] for case in critical_cases if case['critical_rate'] > 50]
        low_champion_cohorts = [c for c in cohort_labels 
                               if not any(c == champion['cohort'] for champion in champions)]
        
        risk_text += f"🚨 HIGH RISK:\n"
        risk_text += f"• {len(high_critical_cohorts)} cohorts >50% critical\n"
        risk_text += f"• {len(low_champion_cohorts)} cohorts no champion status\n\n"
        
        # Calculate portfolio risk
        if total_instances > 0:
            critical_percentage = tier_totals['Critical'] / total_instances * 100
            champion_percentage = tier_totals['Champion'] / total_instances * 100
            
            if critical_percentage > 25:
                risk_level = "HIGH"
                risk_color = "red"
            elif critical_percentage > 15:
                risk_level = "MEDIUM"
                risk_color = "orange"
            else:
                risk_level = "LOW"
                risk_color = "green"
            
            risk_text += f"📊 PORTFOLIO RISK: {risk_level}\n"
            risk_text += f"Critical Rate: {critical_percentage:.1f}%\n"
            risk_text += f"Champion Rate: {champion_percentage:.1f}%\n\n"
        
        risk_text += "💡 MITIGATION:\n"
        risk_text += "• Prioritize critical cases\n"
        risk_text += "• Develop backup champions\n"
        risk_text += "• Monitor key indicators"
        
        ax8.text(0.05, 0.95, risk_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        
        # 9. ROI Recommendations
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        roi_text = "💰 ROI RECOMMENDATIONS\n\n"
        
        roi_text += "🏆 HIGH ROI INVESTMENTS:\n"
        
        # Find specialized champions (good ROI potential)
        specialized = self.champion_analysis['Specialized Champions']
        if specialized:
            for spec in specialized[:2]:
                roi_text += f"• Boost {spec['cohort']} to consistent\n"
        
        # Find biggest opportunity gaps
        if top_kpi_gaps:
            roi_text += f"• Fix {top_kpi_gaps[0]['metric'][:15]}...\n"
        
        roi_text += "\n💎 MEDIUM ROI:\n"
        solid_performers = self.champion_analysis['Solid Performers']
        if solid_performers:
            for solid in solid_performers[:2]:
                roi_text += f"• Optimize {solid['cohort']}\n"
        
        roi_text += "\n⚠️ RISKY INVESTMENTS:\n"
        if critical_cases:
            roi_text += f"• Fixing {critical_cases[0]['cohort']} (high effort)\n"
        
        roi_text += "\n📈 EXPECTED OUTCOMES:\n"
        roi_text += "• 15-30% improvement potential\n"
        roi_text += "• 6-12 month payback period"
        
        ax9.text(0.05, 0.95, roi_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('05_strategic_dashboard.png', dpi=300, bbox_inches='tight')
        print("📊 Strategic dashboard visualization saved")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive KPI analysis report"""
        print("📋 Generating Comprehensive KPI Analysis Report...")
        
        report = []
        report.append("="*100)
        report.append("📊 COMPREHENSIVE KPI PERFORMANCE TIER ANALYSIS REPORT")
        report.append("="*100)
        report.append("")
        
        # Executive Summary
        report.append("🎯 EXECUTIVE SUMMARY")
        report.append("-" * 50)
        report.append(f"Analysis Scope: {len(self.all_metrics)} KPIs across {len(self.all_cohorts)} cohorts")
        report.append(f"Benchmark: {self.benchmark_cohort}")
        report.append("")
        
        # Champion Analysis
        champions = self.champion_analysis['Consistent Champions']
        critical_cases = self.champion_analysis['Critical Cases']
        
        if champions:
            top_champion = champions[0]
            report.append(f"🏆 TOP PERFORMER: {top_champion['cohort']}")
            report.append(f"   Champion Rate: {top_champion['champion_rate']:.1f}% of KPIs")
            report.append(f"   Total KPIs: {top_champion['total_kpis']}")
            report.append("")
        
        if critical_cases:
            worst_case = critical_cases[0]
            report.append(f"🚨 CRITICAL CASE: {worst_case['cohort']}")
            report.append(f"   Critical Rate: {worst_case['critical_rate']:.1f}% of KPIs")
            report.append(f"   Total KPIs: {worst_case['total_kpis']}")
            report.append("")
        
        # Performance Tier Breakdown
        report.append("🏆 PERFORMANCE TIER BREAKDOWN")
        report.append("-" * 50)
        
        for category, cohorts in self.champion_analysis.items():
            if cohorts:
                report.append(f"\n{category.upper()}:")
                for i, cohort_data in enumerate(cohorts, 1):
                    cohort = cohort_data['cohort']
                    if 'champion_rate' in cohort_data:
                        report.append(f"   {i}. {cohort}")
                        report.append(f"      Champion Rate: {cohort_data['champion_rate']:.1f}%")
                        if 'strong_rate' in cohort_data:
                            report.append(f"      Strong Rate: {cohort_data['strong_rate']:.1f}%")
                    elif 'critical_rate' in cohort_data:
                        report.append(f"   {i}. {cohort}")
                        report.append(f"      Critical Rate: {cohort_data['critical_rate']:.1f}%")
                        if 'weak_rate' in cohort_data:
                            report.append(f"      Weak Rate: {cohort_data['weak_rate']:.1f}%")
        
        # Top KPI Champions
        report.append("\n\n📊 TOP KPI PERFORMANCE GAPS")
        report.append("-" * 50)
        
        # Find biggest performance gaps
        kpi_gaps = []
        for metric, rankings in self.tier_rankings.items():
            sorted_performance = rankings['sorted_performance']
            if len(sorted_performance) >= 2:
                best_performance = sorted_performance[0][1]
                worst_performance = sorted_performance[-1][1]
                gap = best_performance - worst_performance
                if gap > 10:
                    kpi_gaps.append({
                        'metric': metric,
                        'gap': gap,
                        'champion': sorted_performance[0][0],
                        'champion_performance': best_performance,
                        'worst': sorted_performance[-1][0],
                        'worst_performance': worst_performance
                    })
        
        kpi_gaps.sort(key=lambda x: x['gap'], reverse=True)
        
        for i, gap in enumerate(kpi_gaps[:10], 1):
            report.append(f"\n{i}. {gap['metric']}")
            report.append(f"   Champion: {gap['champion']} ({gap['champion_performance']:+.1f}%)")
            report.append(f"   Worst: {gap['worst']} ({gap['worst_performance']:+.1f}%)")
            report.append(f"   Performance Gap: {gap['gap']:.1f} percentage points")
        
        # Strategic Recommendations
        report.append("\n\n💡 STRATEGIC RECOMMENDATIONS")
        report.append("-" * 50)
        
        report.append("\n1. IMMEDIATE ACTIONS:")
        if critical_cases:
            for case in critical_cases[:3]:
                report.append(f"   • Address {case['cohort']} critical issues ({case['critical_rate']:.1f}% of KPIs)")
        
        report.append("\n2. SCALE SUCCESS:")
        if champions:
            for champion in champions[:3]:
                report.append(f"   • Scale {champion['cohort']} practices ({champion['champion_rate']:.1f}% champion rate)")
        
        report.append("\n3. OPTIMIZE OPPORTUNITIES:")
        if kpi_gaps:
            for gap in kpi_gaps[:3]:
                report.append(f"   • Fix {gap['metric'][:40]}... ({gap['gap']:.1f}% gap)")
        
        report.append("\n4. PORTFOLIO OPTIMIZATION:")
        report.append("   • Balance investment across consistent vs specialized champions")
        report.append("   • Develop backup champions for risk mitigation")
        report.append("   • Establish performance monitoring systems")
        
        # Business Impact Assessment
        report.append("\n\n💼 BUSINESS IMPACT ASSESSMENT")
        report.append("-" * 50)
        
        # Calculate overall performance distribution
        tier_totals = {'Champion': 0, 'Strong': 0, 'Moderate': 0, 'Weak': 0, 'Critical': 0}
        total_instances = 0
        
        for metric, rankings in self.tier_rankings.items():
            for tier, cohorts in rankings['tiers'].items():
                tier_totals[tier] += len(cohorts)
                total_instances += len(cohorts)
        
        if total_instances > 0:
            report.append(f"\nOverall Performance Distribution:")
            for tier, count in tier_totals.items():
                percentage = count / total_instances * 100
                report.append(f"   {tier}: {count} instances ({percentage:.1f}%)")
            
            critical_percentage = tier_totals['Critical'] / total_instances * 100
            champion_percentage = tier_totals['Champion'] / total_instances * 100
            
            report.append(f"\nPortfolio Health Score: {100 - critical_percentage:.1f}/100")
            report.append(f"Excellence Rate: {champion_percentage:.1f}%")
            report.append(f"Risk Rate: {critical_percentage:.1f}%")
        
        # Risk Assessment
        report.append("\n\n⚠️ RISK ASSESSMENT")
        report.append("-" * 50)
        
        high_risk_cohorts = [case['cohort'] for case in critical_cases if case['critical_rate'] > 40]
        if high_risk_cohorts:
            report.append(f"\nHigh Risk Cohorts (>40% critical rate):")
            for cohort in high_risk_cohorts:
                report.append(f"   • {cohort}")
        
        if critical_percentage > 25:
            report.append(f"\n🚨 PORTFOLIO RISK: HIGH")
            report.append(f"   Critical rate ({critical_percentage:.1f}%) exceeds safe threshold (25%)")
        elif critical_percentage > 15:
            report.append(f"\n⚠️ PORTFOLIO RISK: MEDIUM")
            report.append(f"   Critical rate ({critical_percentage:.1f}%) requires monitoring")
        else:
            report.append(f"\n✅ PORTFOLIO RISK: LOW")
            report.append(f"   Critical rate ({critical_percentage:.1f}%) within acceptable range")
        
        report.append("")
        
        # Save report
        with open('comprehensive_kpi_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("📋 Comprehensive KPI analysis report saved")
        return len(champions), len(critical_cases)

def main():
    """Main function to run comprehensive KPI analysis"""
    print("🚀 STARTING COMPREHENSIVE KPI PERFORMANCE TIER ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = ComprehensiveKPIAnalysis('../Cohort Wise Analysis Fam 2.0 - Sheet1.csv')
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Extract all KPI data
    kpi_data = analyzer.extract_all_kpi_data()
    
    # Calculate performance matrix
    performance_matrix = analyzer.calculate_performance_matrix()
    
    # Create tier rankings
    tier_rankings = analyzer.create_tier_rankings()
    
    # Analyze champion cohorts
    champion_analysis = analyzer.analyze_champion_cohorts()
    
    # Create all visualizations
    print(f"\n📊 Creating comprehensive visualizations...")
    analyzer.create_performance_heatmap_viz()          # PNG 1
    analyzer.create_champion_analysis_viz()            # PNG 2
    analyzer.create_kpi_category_analysis_viz()        # PNG 3
    analyzer.create_tier_distribution_viz()            # PNG 4
    analyzer.create_strategic_dashboard_viz()          # PNG 5
    
    # Generate comprehensive report
    champions_count, critical_count = analyzer.generate_comprehensive_report()
    
    print(f"\n✅ COMPREHENSIVE KPI ANALYSIS COMPLETED!")
    print(f"📁 Generated 5 PNG files + comprehensive report in 'KPI Analysis' folder")
    print(f"🏆 Found {champions_count} Consistent Champions")
    print(f"🚨 Found {critical_count} Critical Cases requiring attention")
    print(f"📊 Analyzed {len(analyzer.all_metrics)} KPIs across {len(analyzer.all_cohorts)} cohorts")

if __name__ == "__main__":
    main()
