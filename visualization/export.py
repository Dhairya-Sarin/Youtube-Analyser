import pandas as pd
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import BarChart, LineChart, Reference
import logging


class DataExporter:
    """Handles data export in various formats"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def export_to_excel(self,
                        results: Dict[str, Any],
                        filename: Optional[str] = None) -> str:
        """Export analysis results to Excel file"""
        try:
            if filename is None:
                filename = f"youtube_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            # Create temporary file if no path specified
            if not os.path.dirname(filename):
                temp_file = tempfile.mktemp(suffix='.xlsx')
                filename = temp_file

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Main data sheet
                if 'data' in results:
                    df = results['data']
                    df.to_excel(writer, sheet_name='Video Data', index=False)
                    self._format_main_sheet(writer.sheets['Video Data'])

                # Channel overview
                if 'channel_info' in results:
                    self._create_overview_sheet(writer, results)

                # Feature summary
                if 'feature_summary' in results:
                    self._create_feature_summary_sheet(writer, results['feature_summary'])

                # Model results
                if 'model_results' in results and results['model_results']:
                    self._create_model_results_sheet(writer, results['model_results'])

                # Insights
                if 'insights' in results:
                    self._create_insights_sheet(writer, results['insights'])

                # Top keywords
                if 'top_keywords' in results:
                    keywords_df = pd.DataFrame(results['top_keywords'])
                    keywords_df.to_excel(writer, sheet_name='Top Keywords', index=False)

            self.logger.info(f"Excel file exported to: {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {e}")
            raise

    def export_to_csv(self,
                      data: pd.DataFrame,
                      filename: Optional[str] = None) -> str:
        """Export DataFrame to CSV"""
        try:
            if filename is None:
                filename = f"youtube_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            data.to_csv(filename, index=False)
            self.logger.info(f"CSV file exported to: {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            raise

    def export_to_json(self,
                       results: Dict[str, Any],
                       filename: Optional[str] = None) -> str:
        """Export results to JSON"""
        try:
            if filename is None:
                filename = f"youtube_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Convert DataFrame to dict for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    json_results[key] = value.to_dict('records')
                elif key == 'analysis_timestamp':
                    json_results[key] = value
                else:
                    json_results[key] = value

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, default=str, ensure_ascii=False)

            self.logger.info(f"JSON file exported to: {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            raise

    def _format_main_sheet(self, worksheet):
        """Format the main data sheet"""
        # Header formatting
        header_font = Font(bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')

        for col in worksheet[1]:
            col.font = header_font
            col.fill = header_fill
            col.alignment = Alignment(horizontal='center')

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    def _create_overview_sheet(self, writer, results: Dict[str, Any]):
        """Create channel overview sheet"""
        channel_info = results.get('channel_info', {})
        data = results.get('data', pd.DataFrame())

        overview_data = {
            'Metric': [
                'Channel Name',
                'Total Videos Analyzed',
                'Date Range',
                'Total Views',
                'Average Views',
                'Max Views',
                'Total Subscribers',
                'Channel Age (days)',
                'Analysis Date'
            ],
            'Value': [
                channel_info.get('title', results.get('channel_name', 'Unknown')),
                len(data),
                f"{data['published_at'].min()} to {data['published_at'].max()}" if not data.empty and 'published_at' in data.columns else 'N/A',
                data['views'].sum() if 'views' in data.columns else 0,
                data['views'].mean() if 'views' in data.columns else 0,
                data['views'].max() if 'views' in data.columns else 0,
                channel_info.get('subscriber_count', 0),
                channel_info.get('channel_age_days', 0),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }

        overview_df = pd.DataFrame(overview_data)
        overview_df.to_excel(writer, sheet_name='Channel Overview', index=False)

    def _create_feature_summary_sheet(self, writer, feature_summary: Dict[str, Any]):
        """Create feature summary sheet"""
        summary_data = []

        # Basic stats
        summary_data.append(['Total Videos', feature_summary.get('total_videos', 0)])
        summary_data.append(['Total Features', feature_summary.get('feature_count', 0)])
        summary_data.append(['Numeric Features', feature_summary.get('numeric_features', 0)])

        # Missing data summary
        missing_data = feature_summary.get('missing_data', {})
        if missing_data:
            summary_data.append(['Features with Missing Data', sum(1 for v in missing_data.values() if v > 0)])
            summary_data.append(['Total Missing Values', sum(missing_data.values())])

        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Feature Summary', index=False)

        # Top correlations if available
        if 'top_view_correlations' in feature_summary:
            correlations = feature_summary['top_view_correlations']
            corr_data = []
            for feature, correlation in correlations.items():
                if feature != 'views':
                    corr_data.append([feature, correlation])

            if corr_data:
                corr_df = pd.DataFrame(corr_data, columns=['Feature', 'Correlation with Views'])
                corr_df.to_excel(writer, sheet_name='Feature Correlations', index=False)

    def _create_model_results_sheet(self, writer, model_results: Dict[str, Any]):
        """Create model results sheet"""
        # Model performance
        performance_data = [
            ['R² Score', model_results.get('r2_score', 0)],
            ['Mean Squared Error', model_results.get('mse', 0)],
            ['Mean Absolute Error', model_results.get('mae', 0)],
            ['Training Samples', model_results.get('training_samples', 0)],
            ['Test Samples', model_results.get('test_samples', 0)],
            ['Model Type', model_results.get('model_type', 'Unknown')]
        ]

        # Cross-validation results
        if 'cv_mean' in model_results:
            performance_data.extend([
                ['CV Mean Score', model_results['cv_mean']],
                ['CV Std Deviation', model_results['cv_std']]
            ])

        performance_df = pd.DataFrame(performance_data, columns=['Metric', 'Value'])
        performance_df.to_excel(writer, sheet_name='Model Performance', index=False)

        # Feature importance
        if 'feature_importance' in model_results:
            importance_data = model_results['feature_importance']
            importance_df = pd.DataFrame(importance_data)
            importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)

    def _create_insights_sheet(self, writer, insights: Dict[str, Any]):
        """Create insights sheet"""
        insights_data = []

        # Performance insights
        if 'performance_insights' in insights:
            perf = insights['performance_insights']
            insights_data.append(['Performance', 'Average Views', perf.get('avg_views', 0)])
            insights_data.append(['Performance', 'Median Views', perf.get('median_views', 0)])

            if 'view_distribution' in perf:
                dist = perf['view_distribution']
                insights_data.append(['Performance', 'Top 20% Avg Views', dist.get('top_20_percent_avg', 0)])
                insights_data.append(['Performance', 'Bottom 20% Avg Views', dist.get('bottom_20_percent_avg', 0)])
                insights_data.append(['Performance', 'Performance Gap Ratio', dist.get('performance_gap', 0)])

        # Content insights
        if 'content_insights' in insights:
            content = insights['content_insights']
            if 'title_patterns' in content:
                title = content['title_patterns']
                insights_data.append(['Content', 'Average Title Length', title.get('avg_title_length', 0)])
                insights_data.append(['Content', 'Optimal Title Length', title.get('optimal_title_length', 0)])

        # Timing insights
        if 'timing_insights' in insights:
            timing = insights['timing_insights']
            insights_data.append(['Timing', 'Optimal Posting Hour', timing.get('optimal_posting_hour', 'N/A')])
            insights_data.append(['Timing', 'Optimal Posting Day', timing.get('optimal_posting_day', 'N/A')])

        # Recommendations
        if 'optimization_recommendations' in insights:
            for i, rec in enumerate(insights['optimization_recommendations']):
                insights_data.append(['Recommendation', f'#{i + 1}', rec])

        if insights_data:
            insights_df = pd.DataFrame(insights_data, columns=['Category', 'Metric', 'Value'])
            insights_df.to_excel(writer, sheet_name='Insights', index=False)
