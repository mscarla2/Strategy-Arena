#!/usr/bin/env python3
"""
ALPHAGENE Chart Visualization System
Interactive charts using Plotly for strategy analysis.
"""

import json
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, TRADING_DAYS_PER_YEAR


# ═══════════════════════════════════════════════════════════════════════════════
# THEME CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class ChartTheme:
    """Dark theme matching terminal UI aesthetic."""
    
    # Colors
    BACKGROUND = "#0d1117"
    PAPER = "#161b22"
    GRID = "#30363d"
    TEXT = "#c9d1d9"
    TEXT_MUTED = "#8b949e"
    
    # Accent colors
    PRIMARY = "#58a6ff"
    SUCCESS = "#3fb950"
    WARNING = "#d29922"
    ERROR = "#f85149"
    ACCENT = "#bc8cff"
    
    # Strategy colors (for comparisons)
    STRATEGY_COLORS = [
        "#58a6ff",  # Blue
        "#3fb950",  # Green
        "#bc8cff",  # Purple
        "#f0883e",  # Orange
        "#ff7b72",  # Red
        "#a5d6ff",  # Light blue
        "#7ee787",  # Light green
        "#d2a8ff",  # Light purple
    ]
    
    # Gene colors
    GENE_COLORS = {
        "momentum": "#58a6ff",
        "value": "#3fb950",
        "quality": "#bc8cff",
        "low_volatility": "#d29922",
        "mean_reversion": "#f85149",
        "size": "#a5d6ff",
        "growth": "#7ee787",
        "dividend": "#d2a8ff",
    }
    
    @classmethod
    def get_gene_color(cls, gene_name: str) -> str:
        """Get color for a gene, with fallback."""
        gene_lower = gene_name.lower()
        for key, color in cls.GENE_COLORS.items():
            if key in gene_lower:
                return color
        # Fallback - hash to consistent color
        idx = hash(gene_name) % len(cls.STRATEGY_COLORS)
        return cls.STRATEGY_COLORS[idx]
    
    @classmethod
    def apply(cls, fig: go.Figure) -> go.Figure:
        """Apply theme to a figure."""
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=cls.PAPER,
            plot_bgcolor=cls.BACKGROUND,
            font=dict(family="Monaco, Consolas, monospace", color=cls.TEXT),
            title_font=dict(size=20, color=cls.PRIMARY),
            legend=dict(
                bgcolor="rgba(22, 27, 34, 0.8)",
                bordercolor=cls.GRID,
                borderwidth=1,
            ),
            xaxis=dict(
                gridcolor=cls.GRID,
                zerolinecolor=cls.GRID,
            ),
            yaxis=dict(
                gridcolor=cls.GRID,
                zerolinecolor=cls.GRID,
            ),
        )
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CHART OUTPUT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ChartOutput:
    """Manages chart saving and display."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or (DATA_DIR / "charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_html(self, fig: go.Figure, filename: str) -> Path:
        """Save figure as interactive HTML."""
        if not filename.endswith('.html'):
            filename += '.html'
        
        filepath = self.output_dir / filename
        fig.write_html(
            filepath,
            include_plotlyjs='cdn',
            full_html=True,
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            }
        )
        return filepath
    
    def save_image(self, fig: go.Figure, filename: str, 
                   format: str = "png", scale: int = 2) -> Path:
        """Save figure as static image."""
        if not filename.endswith(f'.{format}'):
            filename += f'.{format}'
        
        filepath = self.output_dir / filename
        try:
            fig.write_image(filepath, scale=scale)
        except Exception as e:
            print(f"Warning: Could not save image ({e}). Install kaleido: pip install kaleido")
            return None
        return filepath
    
    def show(self, fig: go.Figure, filename: str = None) -> Path:
        """Save and open in browser."""
        if filename is None:
            filename = f"chart_{datetime.now():%Y%m%d_%H%M%S}"
        
        filepath = self.save_html(fig, filename)
        webbrowser.open(f'file://{filepath.absolute()}')
        return filepath


# ═══════════════════════════════════════════════════════════════════════════════
# CORE CHART CLASSES  
# ═══════════════════════════════════════════════════════════════════════════════

class EquityCurveChart:
    """Equity curve visualization with optional benchmarks."""
    
    @staticmethod
    def create(
        equity_curve: pd.Series,
        benchmark_universe: pd.Series = None,
        benchmark_traded: pd.Series = None,
        strategy_name: str = "Strategy",
        show_drawdown: bool = True
    ) -> go.Figure:
        """
        Create equity curve chart with benchmarks.
        
        Args:
            equity_curve: Strategy equity curve (DatetimeIndex -> value)
            benchmark_universe: Equal-weight universe benchmark
            benchmark_traded: Equal-weight of traded tickers benchmark
            strategy_name: Name for legend
            show_drawdown: Whether to show drawdown subplot
        """
        if show_drawdown:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=("Portfolio Value", "Drawdown")
            )
        else:
            fig = go.Figure()
        
        row = 1 if show_drawdown else None
        
        # Normalize to percentage returns for comparison
        def normalize(series):
            return (series / series.iloc[0] - 1) * 100
        
        # Strategy equity
        strategy_norm = normalize(equity_curve)
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=strategy_norm,
                name=strategy_name,
                line=dict(color=ChartTheme.PRIMARY, width=2.5),
                hovertemplate="<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>"
            ),
            row=row, col=1 if show_drawdown else None
        )
        
        # Universe benchmark
        if benchmark_universe is not None:
            bench_norm = normalize(benchmark_universe)
            fig.add_trace(
                go.Scatter(
                    x=benchmark_universe.index,
                    y=bench_norm,
                    name="Universe Equal-Weight",
                    line=dict(color=ChartTheme.TEXT_MUTED, width=1.5, dash='dash'),
                    hovertemplate="<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>"
                ),
                row=row, col=1 if show_drawdown else None
            )
        
        # Traded tickers benchmark
        if benchmark_traded is not None:
            traded_norm = normalize(benchmark_traded)
            fig.add_trace(
                go.Scatter(
                    x=benchmark_traded.index,
                    y=traded_norm,
                    name="Traded Tickers B&H",
                    line=dict(color=ChartTheme.WARNING, width=1.5, dash='dot'),
                    hovertemplate="<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>"
                ),
                row=row, col=1 if show_drawdown else None
            )
        
        # Drawdown subplot
        if show_drawdown:
            cummax = equity_curve.cummax()
            drawdown = (equity_curve - cummax) / cummax * 100
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    fill='tozeroy',
                    name="Drawdown",
                    line=dict(color=ChartTheme.ERROR, width=1),
                    fillcolor="rgba(248, 81, 73, 0.3)",
                    hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.1f}%<extra></extra>"
                ),
                row=2, col=1
            )
            
            fig.update_yaxes(title_text="Return (%)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        else:
            fig.update_yaxes(title_text="Return (%)")
        
        fig.update_layout(
            title=f"📈 {strategy_name} - Equity Curve",
            xaxis_title="Date",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return ChartTheme.apply(fig)


class DrawdownChart:
    """Detailed drawdown analysis."""
    
    @staticmethod
    def create(
        equity_curve: pd.Series,
        strategy_name: str = "Strategy",
        top_n_drawdowns: int = 5
    ) -> go.Figure:
        """Create detailed drawdown chart with annotations."""
        
        # Calculate drawdowns
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax * 100
        
        fig = go.Figure()
        
        # Main drawdown area
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                fill='tozeroy',
                name="Drawdown",
                line=dict(color=ChartTheme.ERROR, width=1.5),
                fillcolor="rgba(248, 81, 73, 0.4)",
                hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>"
            )
        )
        
        # Find and annotate worst drawdowns
        drawdown_periods = DrawdownChart._find_drawdown_periods(equity_curve)
        
        for i, period in enumerate(drawdown_periods[:top_n_drawdowns]):
            # Add vertical span for drawdown period
            fig.add_vrect(
                x0=period['start'],
                x1=period['end'],
                fillcolor="rgba(248, 81, 73, 0.1)",
                layer="below",
                line_width=0,
            )
            
            # Add annotation at trough
            fig.add_annotation(
                x=period['trough_date'],
                y=period['depth'],
                text=f"#{i+1}: {period['depth']:.1f}%<br>{period['duration']}d",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=ChartTheme.TEXT_MUTED,
                font=dict(size=10, color=ChartTheme.TEXT),
                bgcolor=ChartTheme.PAPER,
                bordercolor=ChartTheme.GRID,
                borderwidth=1,
            )
        
        # Add underwater period indicator
        underwater = (drawdown < 0).astype(int)
        underwater_pct = underwater.sum() / len(underwater) * 100
        
        fig.update_layout(
            title=f"📉 {strategy_name} - Drawdown Analysis<br>"
                  f"<sup>Max: {drawdown.min():.1f}% | Underwater: {underwater_pct:.0f}% of time</sup>",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
        )
        
        return ChartTheme.apply(fig)
    
    @staticmethod
    def _find_drawdown_periods(equity: pd.Series) -> List[Dict]:
        """Identify distinct drawdown periods."""
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        
        periods = []
        in_drawdown = False
        start_date = None
        peak_value = None
        
        for date, value in equity.items():
            dd = drawdown[date]
            
            if not in_drawdown and dd < -0.01:  # Start of drawdown
                in_drawdown = True
                start_date = date
                peak_value = cummax[date]
            
            elif in_drawdown and dd >= -0.001:  # Recovery
                in_drawdown = False
                
                # Find trough in this period
                period_dd = drawdown[start_date:date]
                trough_date = period_dd.idxmin()
                trough_value = period_dd.min()
                
                periods.append({
                    'start': start_date,
                    'end': date,
                    'trough_date': trough_date,
                    'depth': trough_value * 100,
                    'duration': (date - start_date).days,
                    'peak_value': peak_value,
                })
        
        # Sort by depth
        periods.sort(key=lambda x: x['depth'])
        return periods


class FitnessEvolutionChart:
    """Visualize fitness across generations."""
    
    @staticmethod
    def create(
        generation_stats: List[Dict],
        show_components: bool = True
    ) -> go.Figure:
        """
        Create fitness evolution chart.
        
        Args:
            generation_stats: List of dicts with keys:
                - generation, avg_fitness, max_fitness, best_ever
                - Optionally: sharpe_component, return_component, consistency_component
        """
        df = pd.DataFrame(generation_stats)
        
        if show_components and 'sharpe_component' in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.6, 0.4],
                subplot_titles=("Fitness Score", "Component Breakdown")
            )
        else:
            fig = go.Figure()
            show_components = False
        
        row = 1 if show_components else None
        
        # Best ever (cumulative max)
        fig.add_trace(
            go.Scatter(
                x=df['generation'],
                y=df['best_ever'],
                name="Best Ever",
                line=dict(color=ChartTheme.SUCCESS, width=3),
                mode='lines+markers',
                marker=dict(size=8, symbol='star'),
                hovertemplate="Gen %{x}<br>Best: %{y:.4f}<extra></extra>"
            ),
            row=row, col=1 if show_components else None
        )
        
        # Max per generation
        fig.add_trace(
            go.Scatter(
                x=df['generation'],
                y=df['max_fitness'],
                name="Generation Max",
                line=dict(color=ChartTheme.PRIMARY, width=2),
                mode='lines+markers',
                hovertemplate="Gen %{x}<br>Max: %{y:.4f}<extra></extra>"
            ),
            row=row, col=1 if show_components else None
        )
        
        # Average fitness
        fig.add_trace(
            go.Scatter(
                x=df['generation'],
                y=df['avg_fitness'],
                name="Generation Avg",
                line=dict(color=ChartTheme.TEXT_MUTED, width=1.5, dash='dash'),
                mode='lines',
                hovertemplate="Gen %{x}<br>Avg: %{y:.4f}<extra></extra>"
            ),
            row=row, col=1 if show_components else None
        )
        
        # Fill between avg and max
        fig.add_trace(
            go.Scatter(
                x=list(df['generation']) + list(df['generation'][::-1]),
                y=list(df['max_fitness']) + list(df['avg_fitness'][::-1]),
                fill='toself',
                fillcolor='rgba(88, 166, 255, 0.1)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=1 if show_components else None
        )
        
        # Component breakdown (stacked area)
        if show_components:
            for component, color in [
                ('sharpe_component', ChartTheme.PRIMARY),
                ('return_component', ChartTheme.SUCCESS),
                ('consistency_component', ChartTheme.ACCENT),
            ]:
                if component in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['generation'],
                            y=df[component],
                            name=component.replace('_', ' ').title(),
                            stackgroup='components',
                            line=dict(width=0.5, color=color),
                            hovertemplate=f"{component}: %{{y:.3f}}<extra></extra>"
                        ),
                        row=2, col=1
                    )
        
        fig.update_layout(
            title="🧬 Fitness Evolution Across Generations",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Generation", dtick=1)
        if show_components:
            fig.update_yaxes(title_text="Fitness Score", row=1, col=1)
            fig.update_yaxes(title_text="Component Value", row=2, col=1)
        else:
            fig.update_yaxes(title_text="Fitness Score")
        
        return ChartTheme.apply(fig)


class GeneCompositionChart:
    """Gene composition visualization."""
    
    @staticmethod
    def create_pie(
        genes: Dict[str, Dict],
        strategy_name: str = "Strategy"
    ) -> go.Figure:
        """Create pie chart of gene weights."""
        
        labels = list(genes.keys())
        weights = [g.get('weight', 1.0) for g in genes.values()]
        colors = [ChartTheme.get_gene_color(g) for g in labels]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=weights,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='inside',
                insidetextorientation='radial',
                hovertemplate="<b>%{label}</b><br>Weight: %{value:.2f}<br>%{percent}<extra></extra>"
            )
        ])
        
        fig.update_layout(
            title=f"🧬 {strategy_name} - Gene Composition",
        )
        
        return ChartTheme.apply(fig)
    
    @staticmethod
    def create_bar(
        genes: Dict[str, Dict],
        strategy_name: str = "Strategy",
        show_params: bool = True
    ) -> go.Figure:
        """Create bar chart with gene details."""
        
        gene_names = list(genes.keys())
        weights = [g.get('weight', 1.0) for g in genes.values()]
        colors = [ChartTheme.get_gene_color(g) for g in gene_names]
        
        # Build hover text with params
        hover_texts = []
        for name, params in genes.items():
            text = f"<b>{name}</b><br>Weight: {params.get('weight', 1.0):.2f}"
            for k, v in params.items():
                if k != 'weight':
                    if isinstance(v, float):
                        text += f"<br>{k}: {v:.2f}"
                    else:
                        text += f"<br>{k}: {v}"
            hover_texts.append(text)
        
        fig = go.Figure(data=[
            go.Bar(
                x=gene_names,
                y=weights,
                marker_color=colors,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
            )
        ])
        
        fig.update_layout(
            title=f"🧬 {strategy_name} - Gene Weights",
            xaxis_title="Gene",
            yaxis_title="Weight",
            showlegend=False,
        )
        
        return ChartTheme.apply(fig)
    
    @staticmethod
    def create_population_heatmap(
        population_genes: List[Dict],
        strategy_names: List[str] = None
    ) -> go.Figure:
        """Create heatmap showing gene presence across population."""
        
        # Get all unique genes
        all_genes = set()
        for genes in population_genes:
            all_genes.update(genes.keys())
        all_genes = sorted(all_genes)
        
        if strategy_names is None:
            strategy_names = [f"S{i+1}" for i in range(len(population_genes))]
        
        # Build matrix
        matrix = []
        for genes in population_genes:
            row = [genes.get(g, {}).get('weight', 0) for g in all_genes]
            matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=all_genes,
            y=strategy_names,
            colorscale=[
                [0, ChartTheme.BACKGROUND],
                [0.5, ChartTheme.PRIMARY],
                [1, ChartTheme.SUCCESS]
            ],
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="🧬 Gene Distribution Across Population",
            xaxis_title="Gene",
            yaxis_title="Strategy",
        )
        
        return ChartTheme.apply(fig)


class StrategyComparisonChart:
    """Compare multiple strategies."""
    
    @staticmethod
    def create(
        equity_curves: Dict[str, pd.Series],
        benchmark: pd.Series = None,
        normalize: bool = True
    ) -> go.Figure:
        """
        Create comparison chart for multiple strategies.
        
        Args:
            equity_curves: Dict mapping strategy_name -> equity_curve
            benchmark: Optional benchmark equity curve
            normalize: Whether to normalize to percentage returns
        """
        fig = go.Figure()
        
        def norm(series):
            if normalize:
                return (series / series.iloc[0] - 1) * 100
            return series
        
        # Add benchmark first (behind)
        if benchmark is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=norm(benchmark),
                    name="Benchmark",
                    line=dict(color=ChartTheme.TEXT_MUTED, width=2, dash='dash'),
                )
            )
        
        # Add strategies
        for i, (name, equity) in enumerate(equity_curves.items()):
            color = ChartTheme.STRATEGY_COLORS[i % len(ChartTheme.STRATEGY_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=norm(equity),
                    name=name[:30],
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{name[:20]}</b><br>%{{x}}<br>Return: %{{y:.1f}}%<extra></extra>"
                )
            )
        
        fig.update_layout(
            title="📊 Strategy Comparison",
            xaxis_title="Date",
            yaxis_title="Return (%)" if normalize else "Portfolio Value",
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(22, 27, 34, 0.8)",
            )
        )
        
        return ChartTheme.apply(fig)


class YearlyReturnsChart:
    """Yearly returns heatmap and bar chart."""
    
    @staticmethod
    def create_heatmap(
        yearly_returns: Dict[str, float],
        strategy_name: str = "Strategy"
    ) -> go.Figure:
        """Create heatmap of yearly returns."""
        
        years = list(yearly_returns.keys())
        returns = list(yearly_returns.values())
        
        # Color scale: red for negative, green for positive
        fig = go.Figure(data=go.Heatmap(
            z=[returns],
            x=years,
            y=[strategy_name],
            colorscale=[
                [0, ChartTheme.ERROR],
                [0.5, ChartTheme.BACKGROUND],
                [1, ChartTheme.SUCCESS]
            ],
            zmid=0,
            text=[[f"{r:.1%}" for r in returns]],
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="<b>%{x}</b><br>Return: %{z:.2%}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"📅 {strategy_name} - Yearly Returns",
            xaxis_title="Year",
            height=200,
        )
        
        return ChartTheme.apply(fig)
    
    @staticmethod
    def create_bar(
        yearly_returns: Dict[str, float],
        strategy_name: str = "Strategy",
        benchmark_returns: Dict[str, float] = None
    ) -> go.Figure:
        """Create bar chart of yearly returns."""
        
        years = list(yearly_returns.keys())
        returns = [r * 100 for r in yearly_returns.values()]
        colors = [ChartTheme.SUCCESS if r >= 0 else ChartTheme.ERROR for r in returns]
        
        fig = go.Figure()
        
        # Strategy bars
        fig.add_trace(
            go.Bar(
                x=years,
                y=returns,
                name=strategy_name,
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>"
            )
        )
        
        # Benchmark line
        if benchmark_returns:
            bench_years = list(benchmark_returns.keys())
            bench_returns = [r * 100 for r in benchmark_returns.values()]
            fig.add_trace(
                go.Scatter(
                    x=bench_years,
                    y=bench_returns,
                    name="Benchmark",
                    mode='lines+markers',
                    line=dict(color=ChartTheme.TEXT_MUTED, width=2, dash='dash'),
                    marker=dict(size=8),
                )
            )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color=ChartTheme.GRID)
        
        fig.update_layout(
            title=f"📅 {strategy_name} - Yearly Returns",
            xaxis_title="Year",
            yaxis_title="Return (%)",
            barmode='group',
        )
        
        return ChartTheme.apply(fig)
    
    @staticmethod
    def create_multi_strategy_heatmap(
        strategies_yearly: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """Create heatmap comparing yearly returns across strategies."""
        
        # Get all years
        all_years = set()
        for yearly in strategies_yearly.values():
            all_years.update(yearly.keys())
        all_years = sorted(all_years)
        
        strategy_names = list(strategies_yearly.keys())
        
        # Build matrix
        matrix = []
        text_matrix = []
        for name in strategy_names:
            yearly = strategies_yearly[name]
            row = [yearly.get(year, 0) for year in all_years]
            matrix.append(row)
            text_matrix.append([f"{r:.1%}" for r in row])
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=all_years,
            y=strategy_names,
            colorscale=[
                [0, ChartTheme.ERROR],
                [0.5, ChartTheme.BACKGROUND],
                [1, ChartTheme.SUCCESS]
            ],
            zmid=0,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2%}<extra></extra>"
        ))
        
        fig.update_layout(
            title="📅 Yearly Returns Comparison",
            xaxis_title="Year",
            yaxis_title="Strategy",
        )
        
        return ChartTheme.apply(fig)


class GenerationBenchmarkChart:
    """Compare generation performance to Buy & Hold."""
    
    @staticmethod
    def create(
        generation_returns: List[Dict],
        benchmark_returns: List[float]
    ) -> go.Figure:
        """
        Compare each generation's best strategy to benchmark.
        
        Args:
            generation_returns: List of dicts with 'generation', 'best_return', 'avg_return'
            benchmark_returns: List of benchmark returns per generation
        """
        df = pd.DataFrame(generation_returns)
        
        fig = go.Figure()
        
        # Best strategy per generation
        fig.add_trace(
            go.Bar(
                x=df['generation'],
                y=[r * 100 for r in df['best_return']],
                name="Best Strategy",
                marker_color=ChartTheme.PRIMARY,
            )
        )
        
        # Average strategy per generation
        fig.add_trace(
            go.Bar(
                x=df['generation'],
                y=[r * 100 for r in df['avg_return']],
                name="Avg Strategy",
                marker_color=ChartTheme.ACCENT,
                opacity=0.7,
            )
        )
        
        # Benchmark line
        fig.add_trace(
            go.Scatter(
                x=df['generation'],
                y=[r * 100 for r in benchmark_returns],
                name="Buy & Hold",
                mode='lines+markers',
                line=dict(color=ChartTheme.WARNING, width=3),
                marker=dict(size=10, symbol='diamond'),
            )
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="solid", line_color=ChartTheme.GRID)
        
        fig.update_layout(
            title="📊 Generation Performance vs Buy & Hold",
            xaxis_title="Generation",
            yaxis_title="Return (%)",
            barmode='group',
            hovermode='x unified',
        )
        
        return ChartTheme.apply(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD (COMBINED CHARTS)
# ═══════════════════════════════════════════════════════════════════════════════

class StrategyDashboard:
    """Combined dashboard for a single strategy."""
    
    @staticmethod
    def create(
        equity_curve: pd.Series,
        genes: Dict[str, Dict],
        yearly_returns: Dict[str, float],
        strategy_name: str = "Strategy",
        benchmark: pd.Series = None,
        metrics: Dict = None
    ) -> go.Figure:
        """Create a comprehensive strategy dashboard."""
        
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{"type": "domain"}, {"type": "xy"}],
                [{"colspan": 2}, None],
            ],
            subplot_titles=(
                "Equity Curve & Drawdown",
                "Gene Composition", "Yearly Returns",
                ""
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            row_heights=[0.45, 0.30, 0.25]
        )
        
        # ═══ ROW 1: Equity Curve ═══
        # Normalize for display
        equity_norm = (equity_curve / equity_curve.iloc[0] - 1) * 100
        
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_norm,
                name="Strategy",
                line=dict(color=ChartTheme.PRIMARY, width=2),
            ),
            row=1, col=1
        )
        
        if benchmark is not None:
            bench_norm = (benchmark / benchmark.iloc[0] - 1) * 100
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=bench_norm,
                    name="Benchmark",
                    line=dict(color=ChartTheme.TEXT_MUTED, width=1.5, dash='dash'),
                ),
                row=1, col=1
            )
        
        # ═══ ROW 2 LEFT: Gene Pie ═══
        labels = list(genes.keys())
        weights = [g.get('weight', 1.0) for g in genes.values()]
        colors = [ChartTheme.get_gene_color(g) for g in labels]
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=weights,
                marker=dict(colors=colors),
                textinfo='label+percent',
                hole=0.4,
                showlegend=False,
            ),
            row=2, col=1
        )
        
        # ═══ ROW 2 RIGHT: Yearly Returns ═══
        years = list(yearly_returns.keys())
        returns = [r * 100 for r in yearly_returns.values()]
        bar_colors = [ChartTheme.SUCCESS if r >= 0 else ChartTheme.ERROR for r in returns]
        
        fig.add_trace(
            go.Bar(
                x=years,
                y=returns,
                marker_color=bar_colors,
                showlegend=False,
            ),
            row=2, col=2
        )
        
        # ═══ ROW 3: Metrics Summary ═══
        if metrics:
            metrics_text = (
                f"<b>Total Return:</b> {metrics.get('total_return', 0):.1%}  |  "
                f"<b>Sharpe:</b> {metrics.get('sharpe_ratio', 0):.2f}  |  "
                f"<b>Max DD:</b> {metrics.get('max_drawdown', 0):.1%}  |  "
                f"<b>Trades:</b> {metrics.get('num_trades', 0)}"
            )
            
            fig.add_annotation(
                x=0.5,
                y=-0.05,
                xref="paper",
                yref="paper",
                text=metrics_text,
                showarrow=False,
                font=dict(size=12, color=ChartTheme.TEXT),
                bgcolor=ChartTheme.PAPER,
                bordercolor=ChartTheme.GRID,
                borderwidth=1,
                borderpad=10,
            )
        
        fig.update_layout(
            title=f"📊 {strategy_name} - Strategy Dashboard",
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return ChartTheme.apply(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE DASHBOARD (FOR ARENA)
# ═══════════════════════════════════════════════════════════════════════════════

class LiveDashboard:
    """Auto-refreshing dashboard for arena runs."""
    
    def __init__(self, output_path: Path = None, refresh_seconds: int = 5):
        self.output_path = output_path or (DATA_DIR / "charts" / "live_dashboard.html")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.refresh_seconds = refresh_seconds
        self.generation_history = []
        self.best_strategy_history = []
    
    def update(
        self,
        generation: int,
        stats: Dict,
        top_strategies: List[Dict] = None,
        benchmark_return: float = None
    ):
        """Update the live dashboard with new generation data."""
        
        self.generation_history.append({
            'generation': generation,
            'avg_fitness': stats.get('avg_fitness', 0),
            'max_fitness': stats.get('max_fitness', 0),
            'best_ever': stats.get('best_ever', 0),
            'benchmark_return': benchmark_return,
        })
        
        if top_strategies:
            self.best_strategy_history.append({
                'generation': generation,
                'strategies': top_strategies[:5]
            })
        
        self._render()
    
    def _render(self):
        """Render the dashboard to HTML."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Fitness Evolution",
                "Top Strategy Genes",
                "Generation Leaderboard",
                "Benchmark Comparison"
            ),
            specs=[
                [{"type": "xy"}, {"type": "domain"}],
                [{"type": "table"}, {"type": "xy"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )
        
        df = pd.DataFrame(self.generation_history)
        
        if not df.empty:
            # ═══ FITNESS EVOLUTION ═══
            fig.add_trace(
                go.Scatter(
                    x=df['generation'],
                    y=df['best_ever'],
                    name="Best Ever",
                    line=dict(color=ChartTheme.SUCCESS, width=3),
                    mode='lines+markers',
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['generation'],
                    y=df['max_fitness'],
                    name="Gen Max",
                    line=dict(color=ChartTheme.PRIMARY, width=2),
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['generation'],
                    y=df['avg_fitness'],
                    name="Gen Avg",
                    line=dict(color=ChartTheme.TEXT_MUTED, width=1, dash='dash'),
                ),
                row=1, col=1
            )
            
            # ═══ BENCHMARK COMPARISON ═══
            if 'benchmark_return' in df.columns and df['benchmark_return'].notna().any():
                fig.add_trace(
                    go.Bar(
                        x=df['generation'],
                        y=df['max_fitness'],
                        name="Strategy",
                        marker_color=ChartTheme.PRIMARY,
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['generation'],
                        y=df['benchmark_return'],
                        name="Benchmark",
                        line=dict(color=ChartTheme.WARNING, width=2),
                        mode='lines+markers',
                    ),
                    row=2, col=2
                )
        
        # ═══ TOP GENES PIE ═══
        if self.best_strategy_history:
            latest = self.best_strategy_history[-1]
            if latest['strategies']:
                # Aggregate genes from top strategies
                gene_weights = {}
                for s in latest['strategies']:
                    for gene_name, params in s.get('genes', {}).items():
                        weight = params.get('weight', 1.0) if isinstance(params, dict) else 1.0
                        gene_weights[gene_name] = gene_weights.get(gene_name, 0) + weight
                
                if gene_weights:
                    labels = list(gene_weights.keys())
                    values = list(gene_weights.values())
                    colors = [ChartTheme.get_gene_color(g) for g in labels]
                    
                    fig.add_trace(
                        go.Pie(
                            labels=labels,
                            values=values,
                            marker=dict(colors=colors),
                            hole=0.4,
                            showlegend=False,
                        ),
                        row=1, col=2
                    )
        
        # ═══ LEADERBOARD TABLE ═══
        if self.best_strategy_history:
            latest = self.best_strategy_history[-1]['strategies']
            
            headers = ['Rank', 'Strategy', 'Fitness', 'Sharpe']
            cells = [
                [f"#{i+1}" for i in range(len(latest))],
                [s.get('name', 'Unknown')[:25] for s in latest],
                [f"{s.get('fitness', 0):.3f}" for s in latest],
                [f"{s.get('sharpe', 0):.2f}" for s in latest],
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=headers,
                        fill_color=ChartTheme.PAPER,
                        font=dict(color=ChartTheme.TEXT),
                        align='left'
                    ),
                    cells=dict(
                        values=cells,
                        fill_color=ChartTheme.BACKGROUND,
                        font=dict(color=ChartTheme.TEXT),
                        align='left'
                    )
                ),
                row=2, col=1
            )
        
        current_gen = self.generation_history[-1]['generation'] if self.generation_history else 0
        fig.update_layout(
            title=f"🧬 ALPHAGENE Live Dashboard - Generation {current_gen}",
            height=800,
        )
        
        fig = ChartTheme.apply(fig)
        
        # Add auto-refresh meta tag
        html_content = fig.to_html(
            include_plotlyjs='cdn',
            full_html=True,
        )
        
        # Inject refresh meta tag
        refresh_tag = f'<meta http-equiv="refresh" content="{self.refresh_seconds}">'
        html_content = html_content.replace('<head>', f'<head>\n{refresh_tag}')
        
        with open(self.output_path, 'w') as f:
            f.write(html_content)
    
    def open_browser(self):
        """Open the dashboard in browser."""
        webbrowser.open(f'file://{self.output_path.absolute()}')
    
    def get_path(self) -> Path:
        return self.output_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN API CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaGeneCharts:
    """Main API for all AlphaGene visualizations."""
    
    def __init__(self, output_dir: Path = None):
        self.output = ChartOutput(output_dir)
        self.live_dashboard: Optional[LiveDashboard] = None
    
    # ─── Individual Charts ───
    
    def equity_curve(self, equity: pd.Series, 
                     benchmark_universe: pd.Series = None,
                     benchmark_traded: pd.Series = None,
                     name: str = "Strategy", show: bool = True) -> go.Figure:
        """Create and optionally show equity curve chart."""
        fig = EquityCurveChart.create(
            equity, benchmark_universe, benchmark_traded, name
        )
        if show:
            self.output.show(fig, f"equity_{name}")
        return fig
    
    def drawdown(self, equity: pd.Series, 
                 name: str = "Strategy", show: bool = True) -> go.Figure:
        """Create and optionally show drawdown chart."""
        fig = DrawdownChart.create(equity, name)
        if show:
            self.output.show(fig, f"drawdown_{name}")
        return fig
    
    def fitness_evolution(self, stats: List[Dict], 
                          show: bool = True) -> go.Figure:
        """Create fitness evolution chart."""
        fig = FitnessEvolutionChart.create(stats)
        if show:
            self.output.show(fig, "fitness_evolution")
        return fig
    
    def gene_composition(self, genes: Dict[str, Dict],
                         name: str = "Strategy",
                         chart_type: str = "pie", 
                         show: bool = True) -> go.Figure:
        """Create gene composition chart (pie or bar)."""
        if chart_type == "pie":
            fig = GeneCompositionChart.create_pie(genes, name)
        else:
            fig = GeneCompositionChart.create_bar(genes, name)
        if show:
            self.output.show(fig, f"genes_{name}")
        return fig
    
    def strategy_comparison(self, equity_curves: Dict[str, pd.Series],
                            benchmark: pd.Series = None,
                            show: bool = True) -> go.Figure:
        """Compare multiple strategies."""
        fig = StrategyComparisonChart.create(equity_curves, benchmark)
        if show:
            self.output.show(fig, "strategy_comparison")
        return fig
    
    def yearly_returns(self, yearly: Dict[str, float],
                       name: str = "Strategy",
                       benchmark: Dict[str, float] = None,
                       show: bool = True) -> go.Figure:
        """Create yearly returns chart."""
        fig = YearlyReturnsChart.create_bar(yearly, name, benchmark)
        if show:
            self.output.show(fig, f"yearly_{name}")
        return fig
    
    def yearly_returns_heatmap(self, strategies_yearly: Dict[str, Dict[str, float]],
                               show: bool = True) -> go.Figure:
        """Create multi-strategy yearly returns heatmap."""
        fig = YearlyReturnsChart.create_multi_strategy_heatmap(strategies_yearly)
        if show:
            self.output.show(fig, "yearly_heatmap")
        return fig
    
    def generation_vs_benchmark(self, gen_returns: List[Dict],
                                 benchmark_returns: List[float],
                                 show: bool = True) -> go.Figure:
        """Compare generations to benchmark."""
        fig = GenerationBenchmarkChart.create(gen_returns, benchmark_returns)
        if show:
            self.output.show(fig, "generation_benchmark")
        return fig
    
    # ─── Dashboards ───
    
    def strategy_dashboard(self, equity: pd.Series, genes: Dict,
                           yearly: Dict, name: str = "Strategy",
                           benchmark: pd.Series = None,
                           metrics: Dict = None,
                           show: bool = True) -> go.Figure:
        """Create comprehensive strategy dashboard."""
        fig = StrategyDashboard.create(
            equity, genes, yearly, name, benchmark, metrics
        )
        if show:
            self.output.show(fig, f"dashboard_{name}")
        return fig
    
    # ─── Live Dashboard ───
    
    def start_live_dashboard(self, refresh_seconds: int = 5) -> LiveDashboard:
        """Start live dashboard for arena."""
        self.live_dashboard = LiveDashboard(
            output_path=self.output.output_dir / "live_dashboard.html",
            refresh_seconds=refresh_seconds
        )
        self.live_dashboard.open_browser()
        return self.live_dashboard
    
    def update_live(self, generation: int, stats: Dict,
                    top_strategies: List[Dict] = None,
                    benchmark_return: float = None):
        """Update live dashboard."""
        if self.live_dashboard:
            self.live_dashboard.update(generation, stats, top_strategies, benchmark_return)


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD CHART FUNCTIONS (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

def create_oil_overlay_chart(
    strategy_returns: pd.Series,
    oil_prices: pd.Series,
    strategy_name: str = "Strategy",
    oil_name: str = "USO (WTI Proxy)",
) -> go.Figure:
    """Dual-axis Plotly chart: strategy cumulative returns + oil price overlay.

    Args:
        strategy_returns: Series of period returns (not cumulative).
        oil_prices: Series of oil ETF prices (e.g. USO).
        strategy_name: Legend label for the strategy line.
        oil_name: Legend label for the oil line.

    Returns:
        Plotly Figure with two y-axes.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Strategy cumulative returns
    if strategy_returns is not None and len(strategy_returns) > 0:
        cum = (1 + strategy_returns).cumprod()
        cum_pct = (cum - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=cum_pct.index,
                y=cum_pct.values,
                name=strategy_name,
                line=dict(color=ChartTheme.PRIMARY, width=3),
                hovertemplate="%{x}<br>Return: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=False,
        )

    # Oil price
    if oil_prices is not None and len(oil_prices) > 0:
        fig.add_trace(
            go.Scatter(
                x=oil_prices.index,
                y=oil_prices.values,
                name=oil_name,
                line=dict(color=ChartTheme.WARNING, width=2, dash="dot"),
                hovertemplate="%{x}<br>Price: $%{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="🛢️ Strategy vs Oil Price",
        template="plotly_dark",
        paper_bgcolor=ChartTheme.PAPER,
        plot_bgcolor=ChartTheme.BACKGROUND,
        font=dict(color=ChartTheme.TEXT),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Cumulative Return (%)", secondary_y=False, gridcolor=ChartTheme.GRID)
    fig.update_yaxes(title_text="Oil Price ($)", secondary_y=True, gridcolor=ChartTheme.GRID)

    return fig


def create_alpha_heatmap(
    period_results: list,
    benchmark_results: list,
) -> go.Figure:
    """Quarterly alpha heatmap — green for positive alpha, red for negative.

    Args:
        period_results: List of dicts with 'period_start' and 'total_return'.
        benchmark_results: List of dicts with 'period_start' and 'total_return'.

    Returns:
        Plotly Figure (annotated heatmap).
    """
    if not period_results or not benchmark_results:
        fig = go.Figure()
        fig.update_layout(title="Alpha Heatmap — No Data", template="plotly_dark",
                          paper_bgcolor=ChartTheme.PAPER, plot_bgcolor=ChartTheme.BACKGROUND)
        return fig

    # Build lookup for benchmark returns keyed by period_start
    bench_map = {b["period_start"]: b["total_return"] for b in benchmark_results}

    # Compute alpha per period and bucket into year / quarter
    rows: Dict[str, Dict[str, float]] = {}  # year -> {quarter_label: alpha}
    for p in sorted(period_results, key=lambda x: x["period_start"]):
        ps = p["period_start"]
        ts = pd.Timestamp(ps)
        year = str(ts.year)
        quarter = f"Q{(ts.month - 1) // 3 + 1}"
        alpha = p["total_return"] - bench_map.get(ps, 0)
        rows.setdefault(year, {})[quarter] = alpha * 100  # percent

    years = sorted(rows.keys())
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    z = []
    text = []
    for y in years:
        row_vals = []
        row_text = []
        for q in quarters:
            val = rows.get(y, {}).get(q, None)
            row_vals.append(val if val is not None else 0)
            row_text.append(f"{val:+.1f}%" if val is not None else "—")
        z.append(row_vals)
        text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=quarters,
            y=years,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, ChartTheme.ERROR], [0.5, ChartTheme.GRID], [1, ChartTheme.SUCCESS]],
            zmid=0,
            hovertemplate="Year %{y} %{x}<br>Alpha: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="📅 Quarterly Alpha vs Benchmark",
        template="plotly_dark",
        paper_bgcolor=ChartTheme.PAPER,
        plot_bgcolor=ChartTheme.BACKGROUND,
        font=dict(color=ChartTheme.TEXT),
        xaxis_title="Quarter",
        yaxis_title="Year",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_feature_importance_bar(features_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top features by weighted score.

    Args:
        features_df: DataFrame with columns 'feature' and 'weighted_score'.

    Returns:
        Plotly Figure.
    """
    if features_df is None or features_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Feature Importance — No Data", template="plotly_dark",
                          paper_bgcolor=ChartTheme.PAPER, plot_bgcolor=ChartTheme.BACKGROUND)
        return fig

    df = features_df.head(20).sort_values("weighted_score", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["weighted_score"],
            y=df["feature"],
            orientation="h",
            marker_color=ChartTheme.PRIMARY,
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="🏆 Top 20 Features by Weighted Score",
        xaxis_title="Weighted Score (usage × avg fitness)",
        yaxis_title="Feature",
        template="plotly_dark",
        paper_bgcolor=ChartTheme.PAPER,
        plot_bgcolor=ChartTheme.BACKGROUND,
        font=dict(color=ChartTheme.TEXT),
        height=max(400, len(df) * 28),
        margin=dict(l=200),
    )
    return fig


def create_feature_category_pie(categories_df: pd.DataFrame) -> go.Figure:
    """Pie chart of feature usage by category.

    Args:
        categories_df: DataFrame with columns 'category' and 'count'.

    Returns:
        Plotly Figure.
    """
    if categories_df is None or categories_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Feature Categories — No Data", template="plotly_dark",
                          paper_bgcolor=ChartTheme.PAPER, plot_bgcolor=ChartTheme.BACKGROUND)
        return fig

    fig = go.Figure(
        go.Pie(
            labels=categories_df["category"],
            values=categories_df["count"],
            hole=0.4,
            marker=dict(colors=ChartTheme.STRATEGY_COLORS[: len(categories_df)]),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        title="📊 Feature Category Breakdown",
        template="plotly_dark",
        paper_bgcolor=ChartTheme.PAPER,
        plot_bgcolor=ChartTheme.BACKGROUND,
        font=dict(color=ChartTheme.TEXT),
    )
    return fig


def create_fitness_evolution_chart(generation_stats: list) -> go.Figure:
    """Line chart showing best / avg / min fitness across generations.

    Args:
        generation_stats: List of dicts with keys 'generation',
            'max_fitness', 'avg_fitness', 'min_fitness'.

    Returns:
        Plotly Figure.
    """
    if not generation_stats:
        fig = go.Figure()
        fig.update_layout(title="Fitness Evolution — No Data", template="plotly_dark",
                          paper_bgcolor=ChartTheme.PAPER, plot_bgcolor=ChartTheme.BACKGROUND)
        return fig

    gens = [s.get("generation", i) for i, s in enumerate(generation_stats)]
    best = [s.get("max_fitness", s.get("best_fitness", 0)) for s in generation_stats]
    avg = [s.get("avg_fitness", 0) for s in generation_stats]
    worst = [s.get("min_fitness", 0) for s in generation_stats]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=best, name="Best", line=dict(color=ChartTheme.SUCCESS, width=3)))
    fig.add_trace(go.Scatter(x=gens, y=avg, name="Average", line=dict(color=ChartTheme.PRIMARY, width=2)))
    fig.add_trace(go.Scatter(x=gens, y=worst, name="Worst", line=dict(color=ChartTheme.ERROR, width=1, dash="dot")))

    fig.update_layout(
        title="🧬 Fitness Evolution",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        template="plotly_dark",
        paper_bgcolor=ChartTheme.PAPER,
        plot_bgcolor=ChartTheme.BACKGROUND,
        font=dict(color=ChartTheme.TEXT),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_charts() -> AlphaGeneCharts:
    """Factory function to create chart instance."""
    return AlphaGeneCharts()


# Quick test
if __name__ == "__main__":
    import numpy as np
    
    # Generate dummy data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    equity = pd.Series(
        100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01),
        index=dates
    )
    
    benchmark = pd.Series(
        100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.008),
        index=dates
    )
    
    genes = {
        "momentum": {"weight": 1.2, "lookback": 252, "skip": 21},
        "value": {"weight": 0.8, "metric": "pe_ratio"},
        "quality": {"weight": 1.0, "roe_threshold": 0.15},
    }
    
    yearly = {"2020": 0.15, "2021": 0.22, "2022": -0.08, "2023": 0.18}
    
    gen_stats = [
        {"generation": i, "avg_fitness": 0.3 + i*0.02, 
         "max_fitness": 0.5 + i*0.03, "best_ever": 0.5 + i*0.03}
        for i in range(10)
    ]
    
    # Test charts
    charts = AlphaGeneCharts()
    
    print("Testing equity curve...")
    charts.equity_curve(equity, benchmark, name="Test Strategy")
    
    print("\nAll tests passed! Charts saved to:", charts.output.output_dir)