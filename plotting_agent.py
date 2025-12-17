"""
Complete Plotly Visualization Agent
Add this file as plotting_agent.py in your project
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from datetime import datetime
import json
import numpy as np

class PlotlyVisualizationAgent:
    """
    Intelligent plotting agent that automatically generates interactive Plotly visualizations
    for every data analysis query
    """
    
    def __init__(self, output_dir="outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def auto_visualize(self, df, query, analysis_context=None):
        """
        Automatically generate the most appropriate visualization based on data and query
        Returns dict with plot info including HTML file path
        """
        if df is None or df.empty:
            return {'success': False, 'msg': 'No data to visualize'}
        
        query_lower = query.lower()
        
        # Determine best plot type
        plot_type = self._determine_plot_type(query_lower, df)
        
        try:
            fig = self._create_visualization(df, plot_type, query_lower)
            
            if fig:
                # Save as interactive HTML
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                html_filename = f"plot_{timestamp}.html"
                html_path = os.path.join(self.output_dir, html_filename)
                
                # Save interactive HTML
                fig.write_html(html_path)
                
                # Also save static PNG for thumbnail
                png_filename = f"plot_{timestamp}.png"
                png_path = os.path.join(self.output_dir, png_filename)
                try:
                    fig.write_image(png_path, width=1200, height=800)
                except:
                    png_filename = None  # kaleido not installed
                
                return {
                    'success': True,
                    'plot_filename': html_filename,
                    'png_filename': png_filename,
                    'plot_type': plot_type,
                    'is_interactive': True
                }
        except Exception as e:
            print(f"Visualization error: {e}")
            return {'success': False, 'msg': str(e)}
        
        return {'success': False, 'msg': 'Could not generate visualization'}
    
    def _determine_plot_type(self, query, df):
        """Intelligently determine plot type from query keywords and data structure"""
        
        numeric_cols = self._find_numeric_columns(df)
        categorical_cols = self._find_categorical_columns(df)
        time_col = self._find_time_column(df)
        
        # Time series detection
        if any(word in query for word in ['trend', 'over time', 'timeline', 'time series', 
                                          'monthly', 'daily', 'yearly', 'weekly']) or time_col:
            return 'line'
        
        # Distribution keywords
        if any(word in query for word in ['distribution', 'histogram', 'frequency', 'spread']):
            return 'histogram'
        
        # Comparison keywords
        if any(word in query for word in ['compare', 'comparison', 'versus', 'vs', 'top', 
                                          'bottom', 'highest', 'lowest', 'by category']):
            return 'bar'
        
        # Proportion/composition keywords
        if any(word in query for word in ['proportion', 'percentage', 'share', 'pie', 
                                          'composition', 'breakdown']):
            return 'pie'
        
        # Correlation/relationship keywords
        if any(word in query for word in ['correlation', 'relationship', 'scatter', 
                                          'between', 'against']):
            return 'scatter'
        
        # Statistical keywords
        if any(word in query for word in ['box', 'outlier', 'quartile', 'median', 'range']):
            return 'box'
        
        # Matrix/heatmap keywords
        if any(word in query for word in ['heatmap', 'heat map', 'matrix', 'correlation matrix']):
            return 'heatmap'
        
        # Area/cumulative keywords
        if any(word in query for word in ['area', 'cumulative', 'stacked', 'filled']):
            return 'area'
        
        # Funnel keywords
        if any(word in query for word in ['funnel', 'conversion', 'pipeline', 'stages']):
            return 'funnel'
        
        # Geographic keywords
        if any(word in query for word in ['map', 'geographic', 'location', 'country', 'state']):
            return 'map'
        
        # Multiple series keywords
        if any(word in query for word in ['multiple', 'all', 'each', 'every']):
            if time_col:
                return 'line'
            return 'bar'
        
        # Default based on data structure
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            if len(df) <= 20:
                return 'bar'
            elif time_col:
                return 'line'
            else:
                return 'bar'
        elif len(numeric_cols) >= 2:
            return 'scatter'
        elif len(numeric_cols) == 1:
            return 'histogram'
        
        return 'bar'  # Default fallback
    
    def _create_visualization(self, df, plot_type, query):
        """Create the appropriate Plotly visualization"""
        
        if plot_type == 'line':
            return self._create_line_chart(df, query)
        elif plot_type == 'bar':
            return self._create_bar_chart(df, query)
        elif plot_type == 'pie':
            return self._create_pie_chart(df, query)
        elif plot_type == 'scatter':
            return self._create_scatter_plot(df, query)
        elif plot_type == 'histogram':
            return self._create_histogram(df, query)
        elif plot_type == 'box':
            return self._create_box_plot(df, query)
        elif plot_type == 'heatmap':
            return self._create_heatmap(df, query)
        elif plot_type == 'area':
            return self._create_area_chart(df, query)
        elif plot_type == 'funnel':
            return self._create_funnel_chart(df, query)
        else:
            return self._create_bar_chart(df, query)
    
    def _create_line_chart(self, df, query):
        """Create interactive line chart"""
        x_col = self._find_time_column(df) or df.columns[0]
        y_cols = self._find_numeric_columns(df)
        
        if not y_cols:
            return None
        
        fig = go.Figure()
        
        # Add traces for each numeric column
        for y_col in y_cols[:5]:  # Limit to 5 lines
            fig.add_trace(go.Scatter(
                x=df[x_col], 
                y=df[y_col],
                mode='lines+markers',
                name=y_col,
                hovertemplate=f'<b>{y_col}</b><br>%{{x}}<br>%{{y:,.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Trend Analysis: {", ".join(y_cols[:5])}',
            xaxis_title=x_col,
            yaxis_title='Values',
            hovermode='x unified',
            template='plotly_white'
        )
        
        self._apply_theme(fig)
        return fig
    
    def _create_bar_chart(self, df, query):
        """Create interactive bar chart"""
        cat_col = self._find_categorical_column(df)
        num_cols = self._find_numeric_columns(df)
        
        if not cat_col or not num_cols:
            return None
        
        # Sort by first numeric column if data is aggregated
        if len(df) <= 50:
            df_sorted = df.sort_values(by=num_cols[0], ascending=False)
        else:
            df_sorted = df.head(20)  # Show top 20 if too many categories
        
        fig = go.Figure()
        
        # Add bar for each numeric column
        for num_col in num_cols[:3]:  # Limit to 3 metrics
            fig.add_trace(go.Bar(
                x=df_sorted[cat_col],
                y=df_sorted[num_col],
                name=num_col,
                text=df_sorted[num_col],
                texttemplate='%{text:,.0f}',
                textposition='outside',
                hovertemplate=f'<b>%{{x}}</b><br>{num_col}: %{{y:,.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'{", ".join(num_cols[:3])} by {cat_col}',
            xaxis_title=cat_col,
            yaxis_title='Values',
            template='plotly_white',
            barmode='group'
        )
        
        self._apply_theme(fig)
        return fig
    
    def _create_pie_chart(self, df, query):
        """Create interactive pie chart"""
        cat_col = self._find_categorical_column(df)
        num_cols = self._find_numeric_columns(df)
        
        if not cat_col or not num_cols:
            return None
        
        # Use first numeric column for values
        num_col = num_cols[0]
        
        # Aggregate if needed and get top categories
        if len(df) > 15:
            df_agg = df.groupby(cat_col)[num_col].sum().reset_index()
            df_agg = df_agg.nlargest(10, num_col)
        else:
            df_agg = df
        
        fig = go.Figure(data=[go.Pie(
            labels=df_agg[cat_col],
            values=df_agg[num_col],
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>',
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f'Distribution of {num_col} by {cat_col}',
            template='plotly_white'
        )
        
        self._apply_theme(fig)
        return fig
    
    def _create_scatter_plot(self, df, query):
        """Create interactive scatter plot"""
        num_cols = self._find_numeric_columns(df)
        
        if len(num_cols) < 2:
            return None
        
        x_col = num_cols[0]
        y_col = num_cols[1]
        
        # Try to find categorical column for color coding
        cat_col = self._find_categorical_column(df)
        
        if cat_col and df[cat_col].nunique() <= 10:
            fig = px.scatter(df, x=x_col, y=y_col, color=cat_col,
                           title=f'{y_col} vs {x_col}',
                           trendline='ols' if len(df) > 10 else None)
        else:
            fig = px.scatter(df, x=x_col, y=y_col,
                           title=f'{y_col} vs {x_col}',
                           trendline='ols' if len(df) > 10 else None)
        
        fig.update_traces(
            hovertemplate=f'<b>{x_col}</b>: %{{x:,.2f}}<br><b>{y_col}</b>: %{{y:,.2f}}<extra></extra>'
        )
        
        self._apply_theme(fig)
        return fig
    
    def _create_histogram(self, df, query):
        """Create interactive histogram"""
        num_cols = self._find_numeric_columns(df)
        
        if not num_cols:
            return None
        
        col = num_cols[0]
        
        fig = go.Figure(data=[go.Histogram(
            x=df[col],
            nbinsx=30,
            name=col,
            hovertemplate='<b>Range</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>'
        )])
        
        # Add mean and median lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                     annotation_text=f"Median: {median_val:.2f}")
        
        fig.update_layout(
            title=f'Distribution of {col}',
            xaxis_title=col,
            yaxis_title='Frequency',
            template='plotly_white'
        )
        
        self._apply_theme(fig)
        return fig
    
    def _create_box_plot(self, df, query):
        """Create interactive box plot"""
        num_cols = self._find_numeric_columns(df)
        cat_col = self._find_categorical_column(df)
        
        if not num_cols:
            return None
        
        fig = go.Figure()
        
        if cat_col and df[cat_col].nunique() <= 20:
            # Grouped box plot
            for num_col in num_cols[:3]:
                fig.add_trace(go.Box(
                    x=df[cat_col],
                    y=df[num_col],
                    name=num_col,
                    boxmean='sd'
                ))
            title = f'Distribution of {", ".join(num_cols[:3])} by {cat_col}'
        else:
            # Simple box plot
            for num_col in num_cols[:5]:
                fig.add_trace(go.Box(
                    y=df[num_col],
                    name=num_col,
                    boxmean='sd'
                ))
            title = f'Distribution of {", ".join(num_cols[:5])}'
        
        fig.update_layout(
            title=title,
            template='plotly_white'
        )
        
        self._apply_theme(fig)
        return fig
    
    def _create_heatmap(self, df, query):
        """Create correlation heatmap"""
        num_cols = self._find_numeric_columns(df)
        
        if len(num_cols) < 2:
            return None
        
        # Calculate correlation matrix
        corr_matrix = df[num_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            template='plotly_white'
        )
        
        self._apply_theme(fig)
        return fig
    
    def _create_area_chart(self, df, query):
        """Create interactive area chart"""
        x_col = self._find_time_column(df) or df.columns[0]
        y_cols = self._find_numeric_columns(df)
        
        if not y_cols:
            return None
        
        fig = go.Figure()
        
        for y_col in y_cols[:5]:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines',
                name=y_col,
                fill='tonexty',
                hovertemplate=f'<b>{y_col}</b><br>%{{x}}<br>%{{y:,.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Area Chart: {", ".join(y_cols[:5])}',
            xaxis_title=x_col,
            yaxis_title='Values',
            hovermode='x unified',
            template='plotly_white'
        )
        
        self._apply_theme(fig)
        return fig
    
    def _create_funnel_chart(self, df, query):
        """Create interactive funnel chart"""
        cat_col = self._find_categorical_column(df)
        num_cols = self._find_numeric_columns(df)
        
        if not cat_col or not num_cols:
            return None
        
        num_col = num_cols[0]
        df_sorted = df.sort_values(by=num_col, ascending=False)
        
        fig = go.Figure(go.Funnel(
            y=df_sorted[cat_col],
            x=df_sorted[num_col],
            textinfo="value+percent initial",
            hovertemplate='<b>%{y}</b><br>Value: %{x:,.0f}<br>%{percentInitial}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Funnel Chart: {num_col} by {cat_col}',
            template='plotly_white'
        )
        
        self._apply_theme(fig)
        return fig
    
    def _find_time_column(self, df):
        """Find datetime/time column"""
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
            if any(word in col.lower() for word in ['date', 'time', 'month', 'year', 'day', 'timestamp']):
                try:
                    pd.to_datetime(df[col])
                    return col
                except:
                    pass
        return None
    
    def _find_numeric_columns(self, df):
        """Find numeric columns"""
        return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    def _find_categorical_columns(self, df):
        """Find categorical columns"""
        return [col for col in df.columns 
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])]
    
    def _find_categorical_column(self, df):
        """Find first suitable categorical column"""
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                if df[col].nunique() <= 100:  # Not too many unique values
                    return col
        return df.columns[0] if len(df.columns) > 0 else None
    
    def _apply_theme(self, fig):
        """Apply consistent theme to all plots"""
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial, sans-serif"
            ),
            margin=dict(t=80, b=60, l=60, r=40)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')