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

class PlatformFeatureParityAnalysis:
    def __init__(self, file_path: str):
        """Initialize platform and feature parity analysis"""
        self.file_path = file_path
        self.df = None
        self.benchmark_cohort = "Combine All"
        
        # Define platform groups
        self.platform_groups = {
            'Android': 'Android',
            'iOS': 'IOS '  # Note: space after IOS in CSV
        }
        
        # Define feature groups
        self.feature_groups = {
            'SLS': 'SLS ',  # Note: space after SLS in CSV
            'DM': 'DM',
            'Bubble': 'Bubble'
        }
        
        # Define payment groups
        self.payment_groups = {
            'PPI': 'PPI',
            'TPAP': 'TPAP',
            'Both': 'Both'
        }
        
        self.all_metrics = []
        self.kpi_data = {}
        self.parity_analysis = {}
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the CSV data"""
        print("📊 Loading platform & feature parity data...")
        
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
        
        # Get all cohorts
        all_cohorts = list(self.platform_groups.values()) + list(self.feature_groups.values()) + list(self.payment_groups.values())
        
        # Clean percentage values and time values for relevant cohorts
        for col in all_cohorts:
            if col in self.df.columns:
                # Convert to string first
                self.df[col] = self.df[col].astype(str)
                # Remove percentage signs, time units, and clean up
                self.df[col] = self.df[col].str.replace('%', '').str.replace('secs', '').str.replace('mins', '').str.replace('sec', '')
                # Convert to numeric, errors='coerce' will convert invalid values to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def extract_kpi_data(self):
        """Extract KPI data for parity analysis"""
        print("📊 Extracting KPI data for parity analysis...")
        
        # Initialize data storage
        self.kpi_data = {}
        
        # Get all cohorts we're analyzing
        all_cohorts = list(self.platform_groups.values()) + list(self.feature_groups.values()) + list(self.payment_groups.values())
        
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
            for cohort in all_cohorts + [self.benchmark_cohort]:
                if cohort in self.df.columns:
                    cohort_value = row[cohort]
                    if pd.notna(cohort_value) and cohort_value != '' and str(cohort_value) != 'nan':
                        try:
                            self.kpi_data[metric_name][cohort] = float(cohort_value)
                        except:
                            continue
        
        print(f"📊 Extracted {len(self.all_metrics)} KPIs for parity analysis")
        return self.kpi_data
    
    def calculate_platform_parity(self):
        """Calculate Android vs iOS parity across all KPIs"""
        print("📱 Calculating Platform Parity (Android vs iOS)...")
        
        android_cohort = self.platform_groups['Android']
        ios_cohort = self.platform_groups['iOS']
        
        platform_parity = {
            'per_kpi': {},
            'summary': {},
            'gaps': []
        }
        
        android_wins = 0
        ios_wins = 0
        total_comparisons = 0
        parity_scores = []
        
        for metric in self.all_metrics:
            if metric in self.kpi_data:
                android_value = self.kpi_data[metric].get(android_cohort, None)
                ios_value = self.kpi_data[metric].get(ios_cohort, None)
                
                if android_value is not None and ios_value is not None and ios_value != 0:
                    # Calculate parity score (1.0 = perfect parity, >1 = Android advantage, <1 = iOS advantage)
                    parity_score = android_value / ios_value
                    
                    # Calculate percentage difference
                    perf_diff = ((android_value - ios_value) / ios_value) * 100
                    
                    platform_parity['per_kpi'][metric] = {
                        'android_value': android_value,
                        'ios_value': ios_value,
                        'parity_score': parity_score,
                        'performance_diff': perf_diff,
                        'winner': 'Android' if android_value > ios_value else 'iOS',
                        'gap_magnitude': abs(perf_diff)
                    }
                    
                    # Track wins
                    if android_value > ios_value:
                        android_wins += 1
                    else:
                        ios_wins += 1
                    
                    total_comparisons += 1
                    parity_scores.append(parity_score)
                    
                    # Track significant gaps
                    if abs(perf_diff) > 50:  # >50% difference
                        platform_parity['gaps'].append({
                            'metric': metric,
                            'winner': 'Android' if android_value > ios_value else 'iOS',
                            'gap': abs(perf_diff),
                            'android_value': android_value,
                            'ios_value': ios_value
                        })
        
        # Sort gaps by magnitude
        platform_parity['gaps'].sort(key=lambda x: x['gap'], reverse=True)
        
        # Calculate summary statistics
        if parity_scores:
            platform_parity['summary'] = {
                'overall_parity_score': np.mean(parity_scores),
                'android_wins': android_wins,
                'ios_wins': ios_wins,
                'total_comparisons': total_comparisons,
                'android_win_rate': android_wins / total_comparisons * 100,
                'ios_win_rate': ios_wins / total_comparisons * 100,
                'significant_gaps': len(platform_parity['gaps']),
                'max_gap': max(platform_parity['gaps'], key=lambda x: x['gap']) if platform_parity['gaps'] else None
            }
        
        self.parity_analysis['platform'] = platform_parity
        return platform_parity
    
    def calculate_feature_parity(self):
        """Calculate SLS vs DM vs Bubble parity"""
        print("🔧 Calculating Feature Parity (SLS vs DM vs Bubble)...")
        
        feature_cohorts = list(self.feature_groups.values())
        
        feature_parity = {
            'per_kpi': {},
            'summary': {},
            'cannibalization': {}
        }
        
        feature_wins = {cohort: 0 for cohort in feature_cohorts}
        total_comparisons = 0
        
        for metric in self.all_metrics:
            if metric in self.kpi_data:
                feature_values = {}
                
                # Get values for all features
                for feature in feature_cohorts:
                    value = self.kpi_data[metric].get(feature, None)
                    if value is not None:
                        feature_values[feature] = value
                
                if len(feature_values) >= 2:  # Need at least 2 features to compare
                    # Find best and worst performers
                    best_feature = max(feature_values, key=feature_values.get)
                    worst_feature = min(feature_values, key=feature_values.get)
                    
                    best_value = feature_values[best_feature]
                    worst_value = feature_values[worst_feature]
                    
                    # Calculate performance gap
                    if worst_value != 0:
                        perf_gap = ((best_value - worst_value) / worst_value) * 100
                    else:
                        perf_gap = 0
                    
                    feature_parity['per_kpi'][metric] = {
                        'values': feature_values,
                        'best_performer': best_feature,
                        'worst_performer': worst_feature,
                        'performance_gap': perf_gap,
                        'variance': np.var(list(feature_values.values()))
                    }
                    
                    # Track wins
                    feature_wins[best_feature] += 1
                    total_comparisons += 1
        
        # Calculate summary
        if total_comparisons > 0:
            feature_parity['summary'] = {
                'feature_wins': feature_wins,
                'total_comparisons': total_comparisons,
                'win_rates': {feature: wins/total_comparisons*100 for feature, wins in feature_wins.items()},
                'dominant_feature': max(feature_wins, key=feature_wins.get)
            }
        
        # Check for cannibalization patterns
        # (Features performing similarly suggests they might be cannibalizing each other)
        similar_performance = []
        for metric, data in feature_parity['per_kpi'].items():
            if data['performance_gap'] < 20:  # Less than 20% difference
                similar_performance.append({
                    'metric': metric,
                    'gap': data['performance_gap'],
                    'features': data['values']
                })
        
        feature_parity['cannibalization'] = {
            'similar_performance_metrics': similar_performance,
            'cannibalization_risk': len(similar_performance) / total_comparisons * 100 if total_comparisons > 0 else 0
        }
        
        self.parity_analysis['features'] = feature_parity
        return feature_parity
    
    def calculate_payment_parity(self):
        """Calculate PPI vs TPAP vs Both parity"""
        print("💳 Calculating Payment Method Parity...")
        
        payment_cohorts = list(self.payment_groups.values())
        
        payment_parity = {
            'per_kpi': {},
            'summary': {},
            'integration_advantage': {}
        }
        
        payment_wins = {cohort: 0 for cohort in payment_cohorts}
        total_comparisons = 0
        
        # Track how often "Both" (integrated) wins
        both_wins = 0
        both_comparisons = 0
        
        for metric in self.all_metrics:
            if metric in self.kpi_data:
                payment_values = {}
                
                # Get values for all payment methods
                for payment in payment_cohorts:
                    value = self.kpi_data[metric].get(payment, None)
                    if value is not None:
                        payment_values[payment] = value
                
                if len(payment_values) >= 2:
                    # Find best and worst performers
                    best_payment = max(payment_values, key=payment_values.get)
                    worst_payment = min(payment_values, key=payment_values.get)
                    
                    best_value = payment_values[best_payment]
                    worst_value = payment_values[worst_payment]
                    
                    # Calculate performance gap
                    if worst_value != 0:
                        perf_gap = ((best_value - worst_value) / worst_value) * 100
                    else:
                        perf_gap = 0
                    
                    payment_parity['per_kpi'][metric] = {
                        'values': payment_values,
                        'best_performer': best_payment,
                        'worst_performer': worst_payment,
                        'performance_gap': perf_gap
                    }
                    
                    # Track wins
                    payment_wins[best_payment] += 1
                    total_comparisons += 1
                    
                    # Track "Both" performance
                    if 'Both' in payment_values:
                        both_comparisons += 1
                        if best_payment == 'Both':
                            both_wins += 1
        
        # Calculate summary
        if total_comparisons > 0:
            payment_parity['summary'] = {
                'payment_wins': payment_wins,
                'total_comparisons': total_comparisons,
                'win_rates': {payment: wins/total_comparisons*100 for payment, wins in payment_wins.items()},
                'dominant_payment': max(payment_wins, key=payment_wins.get)
            }
            
            # Calculate integration advantage
            if both_comparisons > 0:
                both_win_rate = both_wins / both_comparisons * 100
                payment_parity['integration_advantage'] = {
                    'both_win_rate': both_win_rate,
                    'both_wins': both_wins,
                    'both_comparisons': both_comparisons,
                    'integration_advantage': both_win_rate > 50  # True if "Both" wins more than 50%
                }
        
        self.parity_analysis['payments'] = payment_parity
        return payment_parity
    
    def create_platform_parity_viz(self):
        """Create platform parity visualization (PNG 1)"""
        print("📊 Creating Platform Parity Visualization...")
        
        platform_data = self.parity_analysis['platform']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('📱 PLATFORM PARITY ANALYSIS: Android vs iOS', fontsize=18, fontweight='bold')
        
        # 1. Parity Score Distribution
        ax1 = axes[0, 0]
        
        parity_scores = [data['parity_score'] for data in platform_data['per_kpi'].values()]
        
        if parity_scores:
            # Create histogram
            ax1.hist(parity_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Parity')
            ax1.axvline(x=np.mean(parity_scores), color='green', linestyle='-', linewidth=2, 
                       label=f'Avg: {np.mean(parity_scores):.2f}')
            
            ax1.set_title('Parity Score Distribution\n(1.0 = Perfect Parity)', fontweight='bold')
            ax1.set_xlabel('Parity Score (Android/iOS)')
            ax1.set_ylabel('Number of KPIs')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Win Rate Comparison
        ax2 = axes[0, 1]
        
        summary = platform_data['summary']
        win_rates = [summary['android_win_rate'], summary['ios_win_rate']]
        platforms = ['Android', 'iOS']
        colors = ['#90EE90', '#FFB6C1']
        
        bars = ax2.bar(platforms, win_rates, color=colors, alpha=0.8)
        ax2.set_title('Platform Win Rates', fontweight='bold')
        ax2.set_ylabel('Win Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 3. Top Performance Gaps
        ax3 = axes[1, 0]
        
        top_gaps = platform_data['gaps'][:8]  # Top 8 gaps
        
        if top_gaps:
            metric_names = [gap['metric'][:25] + '...' if len(gap['metric']) > 25 else gap['metric'] 
                           for gap in top_gaps]
            gap_values = [gap['gap'] for gap in top_gaps]
            winners = [gap['winner'] for gap in top_gaps]
            
            # Color bars based on winner
            colors = ['#90EE90' if winner == 'Android' else '#FFB6C1' for winner in winners]
            
            bars = ax3.barh(metric_names, gap_values, color=colors, alpha=0.8)
            ax3.set_title('Biggest Performance Gaps', fontweight='bold')
            ax3.set_xlabel('Performance Gap (%)')
            ax3.grid(True, alpha=0.3)
            
            # Add winner labels
            for bar, winner, gap in zip(bars, winners, gap_values):
                width = bar.get_width()
                ax3.text(width + gap*0.02, bar.get_y() + bar.get_height()/2.,
                        f'{winner}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 4. Parity Analysis Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "📱 PLATFORM PARITY SUMMARY\n\n"
        
        if 'summary' in platform_data:
            summary = platform_data['summary']
            summary_text += f"🏆 OVERALL WINNER:\n{summary['android_wins']} Android vs {summary['ios_wins']} iOS wins\n\n"
            
            summary_text += f"📊 PARITY METRICS:\n"
            summary_text += f"• Overall Parity Score: {summary.get('overall_parity_score', 0):.2f}\n"
            summary_text += f"• Android Win Rate: {summary['android_win_rate']:.1f}%\n"
            summary_text += f"• iOS Win Rate: {summary['ios_win_rate']:.1f}%\n"
            summary_text += f"• Significant Gaps: {summary['significant_gaps']}\n\n"
            
            if summary.get('max_gap'):
                max_gap = summary['max_gap']
                summary_text += f"🚨 BIGGEST GAP:\n{max_gap['metric'][:30]}...\n"
                summary_text += f"{max_gap['winner']}: {max_gap['gap']:.1f}% advantage\n\n"
            
            # Determine overall assessment
            if summary['overall_parity_score'] > 1.5:
                assessment = "ANDROID DOMINANCE"
                color = "lightgreen"
            elif summary['overall_parity_score'] < 0.67:
                assessment = "iOS DOMINANCE"
                color = "lightpink"
            else:
                assessment = "BALANCED COMPETITION"
                color = "lightyellow"
            
            summary_text += f"🎯 ASSESSMENT:\n{assessment}"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('01_platform_parity_analysis.png', dpi=300, bbox_inches='tight')
        print("📱 Platform parity visualization saved")
    
    def create_feature_parity_viz(self):
        """Create feature parity visualization (PNG 2)"""
        print("🔧 Creating Feature Parity Visualization...")
        
        feature_data = self.parity_analysis['features']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('🔧 FEATURE PARITY ANALYSIS: SLS vs DM vs Bubble', fontsize=18, fontweight='bold')
        
        # 1. Feature Win Distribution
        ax1 = axes[0, 0]
        
        if 'summary' in feature_data:
            feature_wins = feature_data['summary']['feature_wins']
            features = list(feature_wins.keys())
            wins = list(feature_wins.values())
            
            colors = ['lightblue', 'lightpink', 'lightgreen']
            bars = ax1.bar(features, wins, color=colors[:len(features)], alpha=0.8)
            ax1.set_title('Feature Performance Wins', fontweight='bold')
            ax1.set_ylabel('Number of KPI Wins')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, win in zip(bars, wins):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(win)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 2. Performance Gap Analysis
        ax2 = axes[0, 1]
        
        # Calculate average performance gaps
        gaps = [data['performance_gap'] for data in feature_data['per_kpi'].values()]
        
        if gaps:
            ax2.hist(gaps, bins=15, alpha=0.7, color='orange', edgecolor='black')
            ax2.axvline(x=np.mean(gaps), color='red', linestyle='--', linewidth=2, 
                       label=f'Avg Gap: {np.mean(gaps):.1f}%')
            ax2.set_title('Performance Gap Distribution', fontweight='bold')
            ax2.set_xlabel('Performance Gap (%)')
            ax2.set_ylabel('Number of KPIs')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Cannibalization Risk Analysis
        ax3 = axes[1, 0]
        
        if 'cannibalization' in feature_data:
            cannib_data = feature_data['cannibalization']
            risk_rate = cannib_data['cannibalization_risk']
            
            # Create pie chart for cannibalization risk
            labels = ['Similar Performance\n(Cannibalization Risk)', 'Clear Differentiation']
            sizes = [risk_rate, 100 - risk_rate]
            colors = ['lightcoral', 'lightgreen']
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax3.set_title('Feature Cannibalization Risk', fontweight='bold')
        
        # 4. Feature Analysis Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "🔧 FEATURE ANALYSIS SUMMARY\n\n"
        
        if 'summary' in feature_data:
            summary = feature_data['summary']
            dominant = summary['dominant_feature']
            
            summary_text += f"🏆 DOMINANT FEATURE:\n{dominant}\n\n"
            
            summary_text += f"📊 WIN RATES:\n"
            for feature, rate in summary['win_rates'].items():
                summary_text += f"• {feature}: {rate:.1f}%\n"
            
            if 'cannibalization' in feature_data:
                cannib_rate = feature_data['cannibalization']['cannibalization_risk']
                summary_text += f"\n⚠️ CANNIBALIZATION RISK:\n{cannib_rate:.1f}% of KPIs show similar performance\n\n"
                
                if cannib_rate > 30:
                    summary_text += "🚨 HIGH RISK: Features may be competing\n"
                    summary_text += "Consider consolidation or differentiation"
                elif cannib_rate > 15:
                    summary_text += "⚠️ MODERATE RISK: Monitor feature overlap"
                else:
                    summary_text += "✅ LOW RISK: Good feature differentiation"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('02_feature_parity_analysis.png', dpi=300, bbox_inches='tight')
        print("🔧 Feature parity visualization saved")
    
    def create_payment_parity_viz(self):
        """Create payment method parity visualization (PNG 3)"""
        print("💳 Creating Payment Parity Visualization...")
        
        payment_data = self.parity_analysis['payments']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('💳 PAYMENT METHOD PARITY ANALYSIS: PPI vs TPAP vs Both', fontsize=18, fontweight='bold')
        
        # 1. Payment Method Win Rates
        ax1 = axes[0, 0]
        
        if 'summary' in payment_data:
            payment_wins = payment_data['summary']['payment_wins']
            payments = list(payment_wins.keys())
            wins = list(payment_wins.values())
            
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            bars = ax1.bar(payments, wins, color=colors[:len(payments)], alpha=0.8)
            ax1.set_title('Payment Method Performance Wins', fontweight='bold')
            ax1.set_ylabel('Number of KPI Wins')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, win in zip(bars, wins):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(win)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 2. Integration Advantage Analysis
        ax2 = axes[0, 1]
        
        if 'integration_advantage' in payment_data:
            integ_data = payment_data['integration_advantage']
            both_rate = integ_data['both_win_rate']
            
            # Create gauge-style chart for "Both" performance
            labels = ['"Both" Wins', 'Single Method Wins']
            sizes = [both_rate, 100 - both_rate]
            colors = ['gold', 'lightgray']
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax2.set_title('Integration Advantage\n("Both" vs Single Methods)', fontweight='bold')
        
        # 3. Performance Gap Distribution
        ax3 = axes[1, 0]
        
        # Calculate performance gaps
        gaps = [data['performance_gap'] for data in payment_data['per_kpi'].values()]
        
        if gaps:
            ax3.hist(gaps, bins=15, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(x=np.mean(gaps), color='red', linestyle='--', linewidth=2, 
                       label=f'Avg Gap: {np.mean(gaps):.1f}%')
            ax3.set_title('Payment Method Performance Gaps', fontweight='bold')
            ax3.set_xlabel('Performance Gap (%)')
            ax3.set_ylabel('Number of KPIs')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Payment Analysis Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "💳 PAYMENT ANALYSIS SUMMARY\n\n"
        
        if 'summary' in payment_data:
            summary = payment_data['summary']
            dominant = summary['dominant_payment']
            
            summary_text += f"🏆 TOP PAYMENT METHOD:\n{dominant}\n\n"
            
            summary_text += f"📊 WIN RATES:\n"
            for payment, rate in summary['win_rates'].items():
                summary_text += f"• {payment}: {rate:.1f}%\n"
            
            if 'integration_advantage' in payment_data:
                integ_data = payment_data['integration_advantage']
                both_rate = integ_data['both_win_rate']
                
                summary_text += f"\n🔗 INTEGRATION INSIGHT:\n"
                summary_text += f"\"Both\" wins {both_rate:.1f}% of the time\n\n"
                
                if integ_data['integration_advantage']:
                    summary_text += "✅ INTEGRATION ADVANTAGE CONFIRMED\n"
                    summary_text += "Multi-method users outperform\n"
                    summary_text += "single-method users significantly"
                else:
                    summary_text += "⚠️ NO CLEAR INTEGRATION ADVANTAGE\n"
                    summary_text += "Single methods competitive\n"
                    summary_text += "with multi-method approach"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('03_payment_parity_analysis.png', dpi=300, bbox_inches='tight')
        print("💳 Payment parity visualization saved")
    
    def create_comprehensive_parity_dashboard(self):
        """Create comprehensive parity dashboard (PNG 4)"""
        print("📊 Creating Comprehensive Parity Dashboard...")
        
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('⚖️ COMPREHENSIVE PARITY ANALYSIS DASHBOARD', fontsize=20, fontweight='bold')
        
        # 1. Platform Parity Summary
        ax1 = axes[0, 0]
        
        platform_summary = self.parity_analysis['platform']['summary']
        
        # Platform win comparison
        wins = [platform_summary['android_wins'], platform_summary['ios_wins']]
        platforms = ['Android', 'iOS']
        colors = ['#90EE90', '#FFB6C1']
        
        bars = ax1.bar(platforms, wins, color=colors, alpha=0.8)
        ax1.set_title('Platform Battle\n(Total Wins)', fontweight='bold')
        ax1.set_ylabel('KPI Wins')
        ax1.grid(True, alpha=0.3)
        
        for bar, win in zip(bars, wins):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(win)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 2. Feature Dominance
        ax2 = axes[0, 1]
        
        feature_summary = self.parity_analysis['features']['summary']
        feature_wins = feature_summary['feature_wins']
        
        features = list(feature_wins.keys())
        wins = list(feature_wins.values())
        colors = ['lightblue', 'lightpink', 'lightgreen']
        
        bars = ax2.bar(features, wins, color=colors[:len(features)], alpha=0.8)
        ax2.set_title('Feature Competition\n(Total Wins)', fontweight='bold')
        ax2.set_ylabel('KPI Wins')
        ax2.grid(True, alpha=0.3)
        
        for bar, win in zip(bars, wins):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(win)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Payment Method Performance
        ax3 = axes[0, 2]
        
        payment_summary = self.parity_analysis['payments']['summary']
        payment_wins = payment_summary['payment_wins']
        
        payments = list(payment_wins.keys())
        wins = list(payment_wins.values())
        colors = ['lightblue', 'lightcoral', 'gold']
        
        bars = ax3.bar(payments, wins, color=colors[:len(payments)], alpha=0.8)
        ax3.set_title('Payment Method Battle\n(Total Wins)', fontweight='bold')
        ax3.set_ylabel('KPI Wins')
        ax3.grid(True, alpha=0.3)
        
        for bar, win in zip(bars, wins):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(win)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Platform Win Rate Trend
        ax4 = axes[1, 0]
        
        android_rate = platform_summary['android_win_rate']
        ios_rate = platform_summary['ios_win_rate']
        
        categories = ['Win Rate']
        android_values = [android_rate]
        ios_values = [ios_rate]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, android_values, width, label='Android', color='#90EE90', alpha=0.8)
        bars2 = ax4.bar(x + width/2, ios_values, width, label='iOS', color='#FFB6C1', alpha=0.8)
        
        ax4.set_title('Platform Win Rates', fontweight='bold')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, android_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        for bar, value in zip(bars2, ios_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 5. Feature Cannibalization Risk
        ax5 = axes[1, 1]
        
        if 'cannibalization' in self.parity_analysis['features']:
            cannib_risk = self.parity_analysis['features']['cannibalization']['cannibalization_risk']
            
            # Gauge chart
            labels = ['Cannibalization Risk', 'Clear Differentiation']
            sizes = [cannib_risk, 100 - cannib_risk]
            colors = ['lightcoral', 'lightgreen']
            
            wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax5.set_title('Feature Cannibalization Risk', fontweight='bold')
        
        # 6. Integration Advantage
        ax6 = axes[1, 2]
        
        if 'integration_advantage' in self.parity_analysis['payments']:
            both_rate = self.parity_analysis['payments']['integration_advantage']['both_win_rate']
            
            # Integration advantage visualization
            labels = ['"Both" Advantage', 'Single Method Competitive']
            sizes = [both_rate, 100 - both_rate]
            colors = ['gold', 'lightgray']
            
            wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax6.set_title('Multi-Method Advantage', fontweight='bold')
        
        # 7. Top Gaps Analysis
        ax7 = axes[2, 0]
        ax7.axis('off')
        
        gaps_text = "🚨 TOP PERFORMANCE GAPS\n\n"
        
        platform_gaps = self.parity_analysis['platform']['gaps'][:3]
        gaps_text += "📱 PLATFORM GAPS:\n"
        for gap in platform_gaps:
            gaps_text += f"• {gap['metric'][:20]}...\n  {gap['winner']}: {gap['gap']:.0f}% gap\n"
        
        ax7.text(0.05, 0.95, gaps_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        
        # 8. Champions Summary
        ax8 = axes[2, 1]
        ax8.axis('off')
        
        champions_text = "🏆 PARITY CHAMPIONS\n\n"
        
        # Platform champion
        android_wins = platform_summary['android_wins']
        ios_wins = platform_summary['ios_wins']
        platform_champion = 'Android' if android_wins > ios_wins else 'iOS'
        
        champions_text += f"📱 PLATFORM: {platform_champion}\n"
        champions_text += f"   Wins: {max(android_wins, ios_wins)}/{android_wins + ios_wins}\n\n"
        
        # Feature champion
        feature_champion = feature_summary['dominant_feature']
        feature_wins_count = feature_summary['feature_wins'][feature_champion]
        
        champions_text += f"🔧 FEATURE: {feature_champion}\n"
        champions_text += f"   Wins: {feature_wins_count}\n\n"
        
        # Payment champion
        payment_champion = payment_summary['dominant_payment']
        payment_wins_count = payment_summary['payment_wins'][payment_champion]
        
        champions_text += f"💳 PAYMENT: {payment_champion}\n"
        champions_text += f"   Wins: {payment_wins_count}"
        
        ax8.text(0.05, 0.95, champions_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # 9. Strategic Recommendations
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        strategy_text = "🎯 STRATEGIC ACTIONS\n\n"
        
        # Platform strategy
        if android_wins > ios_wins * 1.5:
            strategy_text += "📱 PLATFORM STRATEGY:\n• Scale Android practices to iOS\n• Investigate Android advantages\n\n"
        elif ios_wins > android_wins * 1.5:
            strategy_text += "📱 PLATFORM STRATEGY:\n• Scale iOS practices to Android\n• Investigate iOS advantages\n\n"
        else:
            strategy_text += "📱 PLATFORM STRATEGY:\n• Maintain competitive balance\n• Focus on platform-specific optimization\n\n"
        
        # Feature strategy
        if 'cannibalization' in self.parity_analysis['features']:
            cannib_risk = self.parity_analysis['features']['cannibalization']['cannibalization_risk']
            if cannib_risk > 30:
                strategy_text += "🔧 FEATURE STRATEGY:\n• Address cannibalization risk\n• Differentiate feature offerings\n\n"
            else:
                strategy_text += "🔧 FEATURE STRATEGY:\n• Maintain feature diversity\n• Scale winning approaches\n\n"
        
        # Payment strategy
        if 'integration_advantage' in self.parity_analysis['payments']:
            if self.parity_analysis['payments']['integration_advantage']['integration_advantage']:
                strategy_text += "💳 PAYMENT STRATEGY:\n• Promote multi-method usage\n• Leverage integration advantage"
            else:
                strategy_text += "💳 PAYMENT STRATEGY:\n• Optimize single methods\n• Improve integration benefits"
        
        ax9.text(0.05, 0.95, strategy_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('04_comprehensive_parity_dashboard.png', dpi=300, bbox_inches='tight')
        print("📊 Comprehensive parity dashboard saved")
    
    def generate_parity_report(self):
        """Generate comprehensive parity analysis report"""
        print("📋 Generating Platform & Feature Parity Report...")
        
        report = []
        report.append("="*100)
        report.append("⚖️ PLATFORM & FEATURE PARITY ANALYSIS REPORT")
        report.append("="*100)
        report.append("")
        
        # Executive Summary
        report.append("🎯 EXECUTIVE SUMMARY")
        report.append("-" * 50)
        
        # Platform summary
        platform_summary = self.parity_analysis['platform']['summary']
        android_wins = platform_summary['android_wins']
        ios_wins = platform_summary['ios_wins']
        
        report.append(f"📱 PLATFORM BATTLE: Android {android_wins} - {ios_wins} iOS")
        
        if android_wins > ios_wins:
            margin = android_wins - ios_wins
            report.append(f"   🏆 Android leads by {margin} KPIs ({platform_summary['android_win_rate']:.1f}% win rate)")
        else:
            margin = ios_wins - android_wins
            report.append(f"   🏆 iOS leads by {margin} KPIs ({platform_summary['ios_win_rate']:.1f}% win rate)")
        
        report.append("")
        
        # Feature summary
        feature_summary = self.parity_analysis['features']['summary']
        dominant_feature = feature_summary['dominant_feature']
        feature_wins = feature_summary['feature_wins'][dominant_feature]
        
        report.append(f"🔧 FEATURE LEADER: {dominant_feature} ({feature_wins} KPI wins)")
        
        if 'cannibalization' in self.parity_analysis['features']:
            cannib_risk = self.parity_analysis['features']['cannibalization']['cannibalization_risk']
            report.append(f"   ⚠️ Cannibalization Risk: {cannib_risk:.1f}%")
        
        report.append("")
        
        # Payment summary
        payment_summary = self.parity_analysis['payments']['summary']
        dominant_payment = payment_summary['dominant_payment']
        payment_wins = payment_summary['payment_wins'][dominant_payment]
        
        report.append(f"💳 PAYMENT LEADER: {dominant_payment} ({payment_wins} KPI wins)")
        
        if 'integration_advantage' in self.parity_analysis['payments']:
            both_rate = self.parity_analysis['payments']['integration_advantage']['both_win_rate']
            report.append(f"   🔗 Integration Advantage: {both_rate:.1f}% win rate for 'Both'")
        
        report.append("")
        
        # Detailed Platform Analysis
        report.append("📱 DETAILED PLATFORM ANALYSIS")
        report.append("-" * 50)
        
        platform_gaps = self.parity_analysis['platform']['gaps']
        
        report.append("🚨 TOP PLATFORM PERFORMANCE GAPS:")
        for i, gap in enumerate(platform_gaps[:5], 1):
            report.append(f"{i}. {gap['metric']}")
            report.append(f"   {gap['winner']}: {gap['android_value']:.1f} vs {gap['ios_value']:.1f} ({gap['gap']:.1f}% gap)")
        
        report.append("")
        
        # Feature Analysis
        report.append("🔧 DETAILED FEATURE ANALYSIS")
        report.append("-" * 50)
        
        report.append("📊 Feature Win Distribution:")
        for feature, wins in feature_summary['feature_wins'].items():
            win_rate = feature_summary['win_rates'][feature]
            report.append(f"   {feature}: {wins} wins ({win_rate:.1f}%)")
        
        if 'cannibalization' in self.parity_analysis['features']:
            cannib_data = self.parity_analysis['features']['cannibalization']
            report.append(f"\n⚠️ Cannibalization Analysis:")
            report.append(f"   Risk Level: {cannib_data['cannibalization_risk']:.1f}%")
            report.append(f"   Similar Performance KPIs: {len(cannib_data['similar_performance_metrics'])}")
        
        report.append("")
        
        # Payment Analysis
        report.append("💳 DETAILED PAYMENT ANALYSIS")
        report.append("-" * 50)
        
        report.append("📊 Payment Method Win Distribution:")
        for payment, wins in payment_summary['payment_wins'].items():
            win_rate = payment_summary['win_rates'][payment]
            report.append(f"   {payment}: {wins} wins ({win_rate:.1f}%)")
        
        if 'integration_advantage' in self.parity_analysis['payments']:
            integ_data = self.parity_analysis['payments']['integration_advantage']
            report.append(f"\n🔗 Integration Analysis:")
            report.append(f"   'Both' Win Rate: {integ_data['both_win_rate']:.1f}%")
            report.append(f"   Integration Advantage: {'YES' if integ_data['integration_advantage'] else 'NO'}")
        
        report.append("")
        
        # Strategic Recommendations
        report.append("💡 STRATEGIC RECOMMENDATIONS")
        report.append("-" * 50)
        
        report.append("1. PLATFORM OPTIMIZATION:")
        if android_wins > ios_wins * 1.2:
            report.append("   • URGENT: Scale Android best practices to iOS")
            report.append("   • Investigate root causes of Android advantages")
            report.append("   • Consider iOS-specific optimization program")
        elif ios_wins > android_wins * 1.2:
            report.append("   • URGENT: Scale iOS best practices to Android")
            report.append("   • Investigate root causes of iOS advantages")
            report.append("   • Consider Android-specific optimization program")
        else:
            report.append("   • Maintain competitive platform balance")
            report.append("   • Focus on platform-specific user experience optimization")
        
        report.append("")
        report.append("2. FEATURE STRATEGY:")
        if 'cannibalization' in self.parity_analysis['features']:
            cannib_risk = self.parity_analysis['features']['cannibalization']['cannibalization_risk']
            if cannib_risk > 30:
                report.append("   • HIGH PRIORITY: Address feature cannibalization")
                report.append("   • Differentiate feature value propositions")
                report.append("   • Consider feature consolidation or repositioning")
            else:
                report.append("   • Maintain healthy feature competition")
                report.append("   • Scale winning feature approaches across portfolio")
        
        report.append("")
        report.append("3. PAYMENT METHOD OPTIMIZATION:")
        if 'integration_advantage' in self.parity_analysis['payments']:
            if self.parity_analysis['payments']['integration_advantage']['integration_advantage']:
                report.append("   • LEVERAGE: Multi-method integration shows clear advantage")
                report.append("   • Promote 'Both' usage to single-method users")
                report.append("   • Invest in integration experience improvements")
            else:
                report.append("   • OPTIMIZE: Single methods remain competitive")
                report.append("   • Improve integration benefits and user experience")
                report.append("   • Consider method-specific optimization strategies")
        
        report.append("")
        
        # Business Impact
        report.append("💼 BUSINESS IMPACT ASSESSMENT")
        report.append("-" * 50)
        
        # Calculate total gaps and opportunities
        total_gaps = len(platform_gaps)
        significant_gaps = len([gap for gap in platform_gaps if gap['gap'] > 100])
        
        report.append(f"📊 Performance Gap Analysis:")
        report.append(f"   Total Performance Gaps: {total_gaps}")
        report.append(f"   Significant Gaps (>100%): {significant_gaps}")
        
        if platform_gaps:
            max_gap = max(platform_gaps, key=lambda x: x['gap'])
            report.append(f"   Largest Gap: {max_gap['gap']:.1f}% in {max_gap['metric']}")
        
        report.append("")
        
        # Risk Assessment
        platform_risk = "HIGH" if significant_gaps > 3 else "MEDIUM" if significant_gaps > 1 else "LOW"
        report.append(f"⚠️ Platform Risk Level: {platform_risk}")
        
        if platform_risk == "HIGH":
            report.append("   URGENT: Multiple critical performance gaps require immediate attention")
        elif platform_risk == "MEDIUM":
            report.append("   MONITOR: Some performance gaps need optimization")
        else:
            report.append("   STABLE: Platforms performing competitively")
        
        report.append("")
        
        # Save report
        with open('platform_feature_parity_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("📋 Platform & feature parity report saved")
        return len(platform_gaps), significant_gaps

def main():
    """Main function to run platform and feature parity analysis"""
    print("🚀 STARTING PLATFORM & FEATURE PARITY ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = PlatformFeatureParityAnalysis('../Cohort Wise Analysis Fam 2.0 - Sheet1.csv')
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Extract KPI data
    kpi_data = analyzer.extract_kpi_data()
    
    # Run parity analyses
    platform_parity = analyzer.calculate_platform_parity()
    feature_parity = analyzer.calculate_feature_parity()
    payment_parity = analyzer.calculate_payment_parity()
    
    # Create visualizations
    print(f"\n📊 Creating parity visualizations...")
    analyzer.create_platform_parity_viz()          # PNG 1
    analyzer.create_feature_parity_viz()           # PNG 2
    analyzer.create_payment_parity_viz()           # PNG 3
    analyzer.create_comprehensive_parity_dashboard()  # PNG 4
    
    # Generate report
    total_gaps, significant_gaps = analyzer.generate_parity_report()
    
    print(f"\n✅ PLATFORM & FEATURE PARITY ANALYSIS COMPLETED!")
    print(f"📁 Generated 4 PNG files + comprehensive report")
    
    # Print key findings
    platform_summary = platform_parity['summary']
    print(f"📱 Platform Battle: Android {platform_summary['android_wins']} - {platform_summary['ios_wins']} iOS")
    print(f"🔧 Feature Leader: {feature_parity['summary']['dominant_feature']}")
    print(f"💳 Payment Leader: {payment_parity['summary']['dominant_payment']}")
    print(f"🚨 Critical Gaps: {significant_gaps} performance gaps >100%")

if __name__ == "__main__":
    main()
