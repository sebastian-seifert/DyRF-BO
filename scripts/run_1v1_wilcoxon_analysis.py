#!/usr/bin/env python3
"""1v1 Wilcoxon Statistical Analysis Suite for CARP-S Benchmark Sweeps.

Computes paired Wilcoxon signed-rank tests, matched-pairs rank-biserial correlation (r_rb),
Cliff's delta, paired Cohen's d_z, Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections
for 1v1 comparisons of custom surrogate/epistemic approaches against the SMAC3 reference baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class TaskWilcoxonResult:
    task_id: str
    n_seeds: int
    mean_cand_cost: float
    sem_cand_cost: float
    mean_base_cost: float
    sem_base_cost: float
    mean_diff: float
    rel_improvement_pct: float
    w_plus: float
    w_minus: float
    wilcoxon_w: float
    p_raw: float
    p_holm: float
    p_bh: float
    r_rb: float
    cliffs_delta: float
    cohens_d: float
    is_significant: bool
    decision: str  # WIN, LOSS, TIE


@dataclass
class AggregateSummary:
    candidate_id: str
    baseline_id: str
    n_tasks: int
    wins: int
    ties: int
    losses: int
    win_rate: float
    macro_w: float
    macro_p: float
    macro_r_rb: float
    macro_cliffs_delta: float
    macro_cohens_d: float
    macro_is_significant: bool
    mean_rank_cand: float
    mean_rank_base: float
    mean_cost_cand: float
    mean_cost_base: float
    global_rel_improvement_pct: float


class StatisticalMathEngine:
    """Core mathematical and statistical calculations."""

    @staticmethod
    def calculate_r_rb(x_cand: np.ndarray, y_base: np.ndarray) -> Tuple[float, float, float]:
        """Computes W+, W-, and matched-pairs Rank-Biserial Correlation r_rb.
        
        Polarity convention:
            diff = x_cand - y_base
            diff < 0 => candidate has lower (better) cost.
            W+ = sum of ranks where diff < 0 (candidate is better).
            W- = sum of ranks where diff > 0 (baseline is better).
            r_rb in [-1.0, 1.0], negative values favor the candidate.
        """
        diff = np.asarray(x_cand, dtype=float) - np.asarray(y_base, dtype=float)
        nonzero = diff[diff != 0]
        if len(nonzero) == 0:
            return 0.0, 0.0, 0.0
            
        abs_diff = np.abs(nonzero)
        ranks = rankdata(abs_diff)
        w_plus = float(np.sum(ranks[nonzero < 0]))   # Sum of ranks where candidate wins (lower cost)
        w_minus = float(np.sum(ranks[nonzero > 0]))  # Sum of ranks where baseline wins
        w_total = w_plus + w_minus
        
        # r_rb = (W- - W+) / (W+ + W-) where positive W+ favors candidate (so r_rb < 0 means candidate is better)
        r_rb = (w_minus - w_plus) / w_total if w_total > 0 else 0.0
        return w_plus, w_minus, float(r_rb)

    @staticmethod
    def safe_paired_wilcoxon(x_cand: np.ndarray, y_base: np.ndarray) -> Tuple[float, float]:
        """SciPy Wilcoxon wrapper with guaranteed zero-difference and degenerate safety."""
        x = np.asarray(x_cand, dtype=float)
        y = np.asarray(y_base, dtype=float)
        if len(x) != len(y) or len(x) < 2:
            return 0.0, 1.0
            
        diff = x - y
        if np.all(diff == 0):
            return 0.0, 1.0
            
        try:
            res = wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
            stat = float(res.statistic) if hasattr(res, "statistic") else 0.0
            pval = float(res.pvalue) if hasattr(res, "pvalue") else 1.0
            if np.isnan(pval):
                pval = 1.0
            return stat, pval
        except Exception:
            return 0.0, 1.0

    @staticmethod
    def calculate_cliffs_delta(x_cand: np.ndarray, y_base: np.ndarray) -> float:
        """Computes Cliff's delta effect size between candidate and baseline distributions."""
        x = np.asarray(x_cand, dtype=float)
        y = np.asarray(y_base, dtype=float)
        n_x, n_y = len(x), len(y)
        if n_x == 0 or n_y == 0:
            return 0.0
        greater = np.sum(x[:, None] > y[None, :])
        less = np.sum(x[:, None] < y[None, :])
        # Negative delta means candidate values are lower (better)
        return float((greater - less) / (n_x * n_y))

    @staticmethod
    def calculate_cohens_d(x_cand: np.ndarray, y_base: np.ndarray) -> float:
        """Computes paired Cohen's d (with fallback to pooled standard deviation)."""
        x = np.asarray(x_cand, dtype=float)
        y = np.asarray(y_base, dtype=float)
        if len(x) == 0 or len(y) == 0:
            return 0.0
            
        mean_diff = float(np.mean(x) - np.mean(y))
        
        # Try paired difference standard deviation first
        if len(x) == len(y) and len(x) >= 2:
            diff = x - y
            std_diff = float(np.std(diff, ddof=1))
            if std_diff > 1e-12:
                return float(mean_diff / std_diff)
                
        # Fallback to pooled standard deviation
        n_x, n_y = len(x), len(y)
        var_x = float(np.var(x, ddof=1)) if n_x > 1 else 0.0
        var_y = float(np.var(y, ddof=1)) if n_y > 1 else 0.0
        if n_x + n_y > 2:
            pooled_var = ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
        else:
            pooled_var = (var_x + var_y) / 2.0
            
        if pooled_var > 1e-12:
            return float(mean_diff / np.sqrt(pooled_var))
            
        if abs(mean_diff) < 1e-12:
            return 0.0
        return float(np.sign(mean_diff))

    @staticmethod
    def apply_corrections(p_vals: List[float]) -> Tuple[List[float], List[float]]:
        """Computes Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) adjustments."""
        p = np.array(p_vals, dtype=float)
        n = len(p)
        if n == 0:
            return [], []
            
        # Holm-Bonferroni Step-Down
        sort_idx = np.argsort(p)
        sorted_p = p[sort_idx]
        adj_holm = np.zeros(n, dtype=float)
        for i, orig in enumerate(sort_idx):
            adj_holm[orig] = min(1.0, sorted_p[i] * (n - i))
        # Ensure monotonicity
        sorted_adj = adj_holm[sort_idx]
        for i in range(1, n):
            sorted_adj[i] = max(sorted_adj[i], sorted_adj[i-1])
        adj_holm[sort_idx] = sorted_adj
        
        # Benjamini-Hochberg (FDR Step-Up)
        adj_bh = np.zeros(n, dtype=float)
        for i, orig in enumerate(sort_idx):
            adj_bh[orig] = min(1.0, sorted_p[i] * n / (i + 1))
        sorted_bh = adj_bh[sort_idx]
        for i in range(n - 2, -1, -1):
            sorted_bh[i] = min(sorted_bh[i], sorted_bh[i+1])
        adj_bh[sort_idx] = sorted_bh
        
        return list(adj_holm), list(adj_bh)


class DataLoader:
    """Loads and extracts final incumbent evaluation metrics from CARP-S parquet files."""

    @staticmethod
    def load_parquet(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found at: {file_path}")
        return pd.read_parquet(file_path)

    @staticmethod
    def extract_terminal_costs(
        df: pd.DataFrame,
        cost_col: Optional[str] = None,
        trials_cap: Optional[int] = None
    ) -> pd.DataFrame:
        """Extracts the final incumbent cost at maximum trials for every (task_id, seed, optimizer_id)."""
        # Resolve cost column
        if cost_col is None:
            for col in ["trial_value__cost_inc", "trial_value__cost_inc_norm", "trial_value__cost"]:
                if col in df.columns:
                    cost_col = col
                    break
        if cost_col not in df.columns:
            raise KeyError(f"Cost column '{cost_col}' not found in DataFrame columns: {df.columns.tolist()}")

        t_col = "n_trials" if "n_trials" in df.columns else "trial_id"
        filtered = df[df[t_col] <= trials_cap] if (trials_cap is not None and t_col in df.columns) else df
        
        # Sort and take last row per (task_id, seed, optimizer_id)
        terminal = (
            filtered.sort_values(t_col)
            .groupby(["task_id", "seed", "optimizer_id"], as_index=False)
            .last()
        )
        return terminal[["task_id", "seed", "optimizer_id", cost_col]]


class PairingEngine:
    """Performs strict seed matching between candidate approach and baseline."""

    @staticmethod
    def pair_seeds(
        df_terminal: pd.DataFrame,
        cand_id: str,
        base_id: str,
        cost_col: Optional[str] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if cost_col is None:
            cost_col = [c for c in df_terminal.columns if "cost" in c][0]

        tasks = sorted(df_terminal["task_id"].unique().tolist())
        paired_dict = {}

        for task in tasks:
            task_df = df_terminal[df_terminal["task_id"] == task]
            cand_series = task_df[task_df["optimizer_id"] == cand_id].set_index("seed")[cost_col]
            base_series = task_df[task_df["optimizer_id"] == base_id].set_index("seed")[cost_col]
            
            common_seeds = sorted(list(set(cand_series.index).intersection(set(base_series.index))))
            if len(common_seeds) > 0:
                cand_vals = cand_series.loc[common_seeds].to_numpy(dtype=float)
                base_vals = base_series.loc[common_seeds].to_numpy(dtype=float)
                paired_dict[task] = (cand_vals, base_vals)

        return paired_dict


class StatisticalAnalysisEngine:
    """Orchestrates task-level and macro-level 1v1 Wilcoxon comparisons."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def analyze_1v1(
        self,
        paired_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        cand_id: str,
        base_id: str
    ) -> Tuple[List[TaskWilcoxonResult], AggregateSummary]:
        task_entries = []
        raw_p_list = []

        for task_id, (cand_vals, base_vals) in paired_dict.items():
            n_seeds = len(cand_vals)
            w_stat, p_val = StatisticalMathEngine.safe_paired_wilcoxon(cand_vals, base_vals)
            w_plus, w_minus, r_rb = StatisticalMathEngine.calculate_r_rb(cand_vals, base_vals)
            delta = StatisticalMathEngine.calculate_cliffs_delta(cand_vals, base_vals)
            d_z = StatisticalMathEngine.calculate_cohens_d(cand_vals, base_vals)

            mean_cand = float(np.mean(cand_vals))
            sem_cand = float(np.std(cand_vals, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0
            mean_base = float(np.mean(base_vals))
            sem_base = float(np.std(base_vals, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0
            
            mean_diff = mean_cand - mean_base
            rel_imp = ((mean_base - mean_cand) / abs(mean_base) * 100.0) if mean_base != 0 else 0.0

            raw_p_list.append(p_val)
            task_entries.append({
                "task_id": task_id,
                "n_seeds": n_seeds,
                "mean_cand_cost": mean_cand,
                "sem_cand_cost": sem_cand,
                "mean_base_cost": mean_base,
                "sem_base_cost": sem_base,
                "mean_diff": mean_diff,
                "rel_improvement_pct": rel_imp,
                "w_plus": w_plus,
                "w_minus": w_minus,
                "wilcoxon_w": w_stat,
                "p_raw": p_val,
                "r_rb": r_rb,
                "cliffs_delta": delta,
                "cohens_d": d_z,
                "cand_vals": cand_vals,
                "base_vals": base_vals
            })

        p_holm_list, p_bh_list = StatisticalMathEngine.apply_corrections(raw_p_list)

        task_results = []
        wins, ties, losses = 0, 0, 0
        cand_means, base_means = [], []
        cand_ranks, base_ranks = [], []

        for i, entry in enumerate(task_entries):
            p_raw = entry["p_raw"]
            p_holm = p_holm_list[i]
            p_bh = p_bh_list[i]
            r_rb = entry["r_rb"]
            is_sig = bool(p_raw < self.alpha)

            if is_sig and r_rb < 0:
                decision = "WIN"
                wins += 1
            elif is_sig and r_rb > 0:
                decision = "LOSS"
                losses += 1
            else:
                decision = "TIE"
                ties += 1

            m_cand = entry["mean_cand_cost"]
            m_base = entry["mean_base_cost"]
            cand_means.append(m_cand)
            base_means.append(m_base)

            if m_cand < m_base:
                cand_ranks.append(1.0)
                base_ranks.append(2.0)
            elif m_cand > m_base:
                cand_ranks.append(2.0)
                base_ranks.append(1.0)
            else:
                cand_ranks.append(1.5)
                base_ranks.append(1.5)

            task_results.append(TaskWilcoxonResult(
                task_id=entry["task_id"],
                n_seeds=entry["n_seeds"],
                mean_cand_cost=m_cand,
                sem_cand_cost=entry["sem_cand_cost"],
                mean_base_cost=m_base,
                sem_base_cost=entry["sem_base_cost"],
                mean_diff=entry["mean_diff"],
                rel_improvement_pct=entry["rel_improvement_pct"],
                w_plus=entry["w_plus"],
                w_minus=entry["w_minus"],
                wilcoxon_w=entry["wilcoxon_w"],
                p_raw=p_raw,
                p_holm=p_holm,
                p_bh=p_bh,
                r_rb=r_rb,
                cliffs_delta=entry["cliffs_delta"],
                cohens_d=entry["cohens_d"],
                is_significant=is_sig,
                decision=decision
            ))

        # Macro Cross-Task Statistics (T tasks)
        macro_w, macro_p = StatisticalMathEngine.safe_paired_wilcoxon(np.array(cand_means), np.array(base_means))
        _, _, macro_r_rb = StatisticalMathEngine.calculate_r_rb(np.array(cand_means), np.array(base_means))
        macro_is_sig = bool(macro_p < self.alpha)

        macro_cliffs = float(np.mean([t.cliffs_delta for t in task_results])) if task_results else 0.0
        macro_cohens = float(np.mean([t.cohens_d for t in task_results])) if task_results else 0.0

        global_mean_cand = float(np.mean(cand_means)) if cand_means else 0.0
        global_mean_base = float(np.mean(base_means)) if base_means else 0.0
        global_rel_imp = ((global_mean_base - global_mean_cand) / abs(global_mean_base) * 100.0) if global_mean_base != 0 else 0.0

        mean_rank_cand = float(np.mean(cand_ranks)) if cand_ranks else 1.5
        mean_rank_base = float(np.mean(base_ranks)) if base_ranks else 1.5

        agg = AggregateSummary(
            candidate_id=cand_id,
            baseline_id=base_id,
            n_tasks=len(task_results),
            wins=wins,
            ties=ties,
            losses=losses,
            win_rate=(wins / len(task_results) * 100.0) if task_results else 0.0,
            macro_w=macro_w,
            macro_p=macro_p,
            macro_r_rb=macro_r_rb,
            macro_cliffs_delta=macro_cliffs,
            macro_cohens_d=macro_cohens,
            macro_is_significant=macro_is_sig,
            mean_rank_cand=mean_rank_cand,
            mean_rank_base=mean_rank_base,
            mean_cost_cand=global_mean_cand,
            mean_cost_base=global_mean_base,
            global_rel_improvement_pct=global_rel_imp
        )

        return task_results, agg


class ReportExporter:
    """Generates formatted reports across terminal, Markdown, LaTeX, and CSV formats."""

    def __init__(self, output_dir: str = "results/sweep_1v1_analysis/report_1v1_sweeps"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def render_rich_table(self, results: Dict[str, Tuple[List[TaskWilcoxonResult], AggregateSummary]]) -> str:
        """Renders rich formatted tables and returns the text output string."""
        if not RICH_AVAILABLE:
            lines = []
            for cand_id, (tasks, agg) in results.items():
                lines.append(f"Approach: {cand_id} vs {agg.baseline_id}")
                for t in tasks:
                    lines.append(f"  {t.task_id}: Cand={t.mean_cand_cost:.4f} Base={t.mean_base_cost:.4f} [{t.decision}]")
            return "\n".join(lines)

        string_io = io.StringIO()
        console = Console(record=True, width=120, file=string_io)
        for cand_id, (tasks, agg) in results.items():
            table = Table(
                title=f"1v1 Head-to-Head Wilcoxon Signed-Rank Test\nCandidate: [bold cyan]{cand_id}[/bold cyan] vs Baseline: [bold yellow]{agg.baseline_id}[/bold yellow] (α=0.05)",
                box=box.ROUNDED,
                header_style="bold magenta",
                show_footer=True
            )
            table.add_column("Task ID", style="cyan", no_wrap=True)
            table.add_column("Cand Mean ± SEM", justify="right")
            table.add_column("Base Mean ± SEM", justify="right")
            table.add_column("Diff (Δ)", justify="right")
            table.add_column("r_rb", justify="right")
            table.add_column("p (raw)", justify="right")
            table.add_column("p (Holm)", justify="right")
            table.add_column("Sig (α=0.05)", justify="center")
            table.add_column("Verdict", justify="center")

            for t in tasks:
                cand_str = f"{t.mean_cand_cost:.4f} ± {t.sem_cand_cost:.4f}"
                base_str = f"{t.mean_base_cost:.4f} ± {t.sem_base_cost:.4f}"
                diff_str = f"{t.mean_diff:>+0.4f}"
                r_rb_str = f"{t.r_rb:>+0.2f}"
                p_raw_str = f"{t.p_raw:.4f}"
                p_holm_str = f"{t.p_holm:.4f}"
                sig_str = "[bold green]YES[/bold green]" if t.is_significant else "[dim]NO[/dim]"

                if t.decision == "WIN":
                    verdict_str = "[bold green]WIN[/bold green]"
                elif t.decision == "LOSS":
                    verdict_str = "[bold red]LOSS[/bold red]"
                else:
                    verdict_str = "[bold yellow]TIE[/bold yellow]"

                task_clean = t.task_id.replace("subset_yahpo_", "").replace("subset_hpobench_", "")
                table.add_row(
                    task_clean,
                    cand_str,
                    base_str,
                    diff_str,
                    r_rb_str,
                    p_raw_str,
                    p_holm_str,
                    sig_str,
                    verdict_str
                )

            console.print(table)
            
            macro_sig = "[bold green]YES[/bold green]" if agg.macro_is_significant else "[dim]NO[/dim]"
            summary_panel = Panel(
                f"[bold]Aggregate Summary ({agg.n_tasks} Tasks):[/bold]\n"
                f"  • Record: [bold green]{agg.wins} Wins[/bold green] | [bold yellow]{agg.ties} Ties[/bold yellow] | [bold red]{agg.losses} Losses[/bold red] (Win Rate: [bold cyan]{agg.win_rate:.1f}%[/bold cyan])\n"
                f"  • Mean Ranks: Candidate = [bold cyan]{agg.mean_rank_cand:.2f}[/bold cyan] vs Baseline = [bold yellow]{agg.mean_rank_base:.2f}[/bold yellow]\n"
                f"  • Macro Wilcoxon: W = {agg.macro_w:.1f}, p = {agg.macro_p:.4f} (Sig: {macro_sig}), Macro r_rb = {agg.macro_r_rb:>+0.3f}\n"
                f"  • Global Costs: Cand = {agg.mean_cost_cand:.4f} vs Base = {agg.mean_cost_base:.4f} (Rel Improvement: {agg.global_rel_improvement_pct:>+0.2f}%)",
                title=f"Summary for {cand_id}",
                border_style="green" if agg.wins > agg.losses else ("red" if agg.losses > agg.wins else "yellow")
            )
            console.print(summary_panel)
            console.print()

        return console.export_text()

    def print_terminal_report(self, results: Dict[str, Tuple[List[TaskWilcoxonResult], AggregateSummary]]):
        """Prints a rich, human-readable terminal table for each approach."""
        if RICH_AVAILABLE:
            console = Console(width=120)
            for cand_id, (tasks, agg) in results.items():
                table = Table(
                    title=f"1v1 Head-to-Head Wilcoxon Signed-Rank Test\nCandidate: [bold cyan]{cand_id}[/bold cyan] vs Baseline: [bold yellow]{agg.baseline_id}[/bold yellow] (α=0.05)",
                    box=box.ROUNDED,
                    header_style="bold magenta"
                )
                table.add_column("Task ID", style="cyan", no_wrap=True)
                table.add_column("Cand Mean ± SEM", justify="right")
                table.add_column("Base Mean ± SEM", justify="right")
                table.add_column("Diff (Δ)", justify="right")
                table.add_column("r_rb", justify="right")
                table.add_column("p (raw)", justify="right")
                table.add_column("p (Holm)", justify="right")
                table.add_column("Sig (α=0.05)", justify="center")
                table.add_column("Verdict", justify="center")

                for t in tasks:
                    cand_str = f"{t.mean_cand_cost:.4f} ± {t.sem_cand_cost:.4f}"
                    base_str = f"{t.mean_base_cost:.4f} ± {t.sem_base_cost:.4f}"
                    diff_str = f"{t.mean_diff:>+0.4f}"
                    r_rb_str = f"{t.r_rb:>+0.2f}"
                    p_raw_str = f"{t.p_raw:.4f}"
                    p_holm_str = f"{t.p_holm:.4f}"
                    sig_str = "[bold green]YES[/bold green]" if t.is_significant else "[dim]NO[/dim]"

                    if t.decision == "WIN":
                        verdict_str = "[bold green]WIN[/bold green]"
                    elif t.decision == "LOSS":
                        verdict_str = "[bold red]LOSS[/bold red]"
                    else:
                        verdict_str = "[bold yellow]TIE[/bold yellow]"

                    task_clean = t.task_id.replace("subset_yahpo_", "").replace("subset_hpobench_", "")
                    table.add_row(
                        task_clean,
                        cand_str,
                        base_str,
                        diff_str,
                        r_rb_str,
                        p_raw_str,
                        p_holm_str,
                        sig_str,
                        verdict_str
                    )

                console.print(table)
                macro_sig = "[bold green]YES[/bold green]" if agg.macro_is_significant else "[dim]NO[/dim]"
                summary_panel = Panel(
                    f"[bold]Aggregate Summary ({agg.n_tasks} Tasks):[/bold]\n"
                    f"  • Record: [bold green]{agg.wins} Wins[/bold green] | [bold yellow]{agg.ties} Ties[/bold yellow] | [bold red]{agg.losses} Losses[/bold red] (Win Rate: [bold cyan]{agg.win_rate:.1f}%[/bold cyan])\n"
                    f"  • Mean Ranks: Candidate = [bold cyan]{agg.mean_rank_cand:.2f}[/bold cyan] vs Baseline = [bold yellow]{agg.mean_rank_base:.2f}[/bold yellow]\n"
                    f"  • Macro Wilcoxon: W = {agg.macro_w:.1f}, p = {agg.macro_p:.4f} (Sig: {macro_sig}), Macro r_rb = {agg.macro_r_rb:>+0.3f}\n"
                    f"  • Global Costs: Cand = {agg.mean_cost_cand:.4f} vs Base = {agg.mean_cost_base:.4f} (Rel Improvement: {agg.global_rel_improvement_pct:>+0.2f}%)",
                    title=f"Summary for {cand_id}",
                    border_style="green" if agg.wins > agg.losses else ("red" if agg.losses > agg.wins else "yellow")
                )
                console.print(summary_panel)
                console.print()
        else:
            for cand_id, (tasks, agg) in results.items():
                print("\n" + "=" * 105)
                print(f" 1v1 HEAD-TO-HEAD WILCOXON ANALYSIS: {cand_id}")
                print(f" Reference Baseline: {agg.baseline_id} (alpha = 0.05, two-sided)")
                print("=" * 105)
                
                header = (
                    f"{'Task ID':<32} | {'Candidate Cost':<18} | {'Baseline Cost':<18} | "
                    f"{'r_rb':<7} | {'p (raw)':<8} | {'p (Holm)':<8} | {'Sig (α=0.05)':<11} | {'Verdict':<7}"
                )
                print(header)
                print("-" * 105)

                for t in tasks:
                    cand_str = f"{t.mean_cand_cost:.4f} ± {t.sem_cand_cost:.4f}"
                    base_str = f"{t.mean_base_cost:.4f} ± {t.sem_base_cost:.4f}"
                    sig_str = "YES (p<0.05)" if t.is_significant else "NO"
                    verdict_str = f"{t.decision:<7}"
                    task_clean = t.task_id.replace("subset_yahpo_", "").replace("subset_hpobench_", "")
                    print(
                        f"{task_clean:<32} | {cand_str:<18} | {base_str:<18} | "
                        f"{t.r_rb:>+6.2f}  | {t.p_raw:<8.4f} | {t.p_holm:<8.4f} | {sig_str:<11} | {verdict_str}"
                    )

                print("-" * 105)
                macro_sig = "YES" if agg.macro_is_significant else "NO"
                print(f" GLOBAL AGGREGATE SUMMARY ({agg.n_tasks} Tasks):")
                print(f"   -> Win / Tie / Loss:   {agg.wins} Wins | {agg.ties} Ties | {agg.losses} Losses")
                print(f"   -> Win Rate:           {agg.win_rate:.1f}%")
                print(f"   -> Macro Wilcoxon W:   {agg.macro_w:.1f} (p = {agg.macro_p:.4f}, Significant: {macro_sig})")
                print(f"   -> Macro Effect Size:  r_rb = {agg.macro_r_rb:>+0.3f}")
                print(f"   -> Global Cost Mean:   Cand = {agg.mean_cost_cand:.4f} vs Base = {agg.mean_cost_base:.4f} (Rel Imp: {agg.global_rel_improvement_pct:>+0.2f}%)")
                print("=" * 105)

    def export_markdown(self, results: Dict[str, Tuple[List[TaskWilcoxonResult], AggregateSummary]]) -> str:
        md_file = os.path.join(self.output_dir, "1v1_wilcoxon_report.md")
        lines = [
            "# CARP-S 1v1 Paired Wilcoxon Signed-Rank Evaluation Report",
            "",
            "**Significance Criterion:** Two-sided Paired Wilcoxon Signed-Rank test with $\\alpha = 0.05$.",
            "**Effect Size:** Matched-pairs Rank-Biserial Correlation $r_{\\text{rb}} \\in [-1.0, 1.0]$ (negative values indicate lower objective cost for the candidate).",
            "",
            "## 1. Global Cross-Task Aggregate Summary",
            "",
            "| Candidate Approach | Tasks | Wins | Ties | Losses | Win Rate (%) | Mean Rank (Cand / Base) | Global Wilcoxon W | Global p-value | Global Sig (α=0.05) | Macro $r_{\\text{rb}}$ | Rel Imp (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        for cand_id, (_, agg) in results.items():
            sig_str = "**YES**" if agg.macro_is_significant else "NO"
            lines.append(
                f"| `{cand_id}` | {agg.n_tasks} | **{agg.wins}** | {agg.ties} | {agg.losses} | "
                f"{agg.win_rate:.1f}% | {agg.mean_rank_cand:.2f} / {agg.mean_rank_base:.2f} | {agg.macro_w:.1f} | {agg.macro_p:.4f} | {sig_str} | "
                f"{agg.macro_r_rb:>+0.3f} | {agg.global_rel_improvement_pct:>+0.2f}% |"
            )

        lines.extend(["", "## 2. Per-Approach & Per-Task Breakdown", ""])

        for cand_id, (tasks, agg) in results.items():
            lines.append(f"### Approach: `{cand_id}` vs `{agg.baseline_id}`")
            lines.append("")
            lines.append(
                "| Task ID | Seeds | Candidate Cost (Mean ± SEM) | Baseline Cost (Mean ± SEM) | $r_{\\text{rb}}$ | $p_{\\text{raw}}$ | $p_{\\text{Holm}}$ | Significant (α=0.05) | Verdict |"
            )
            lines.append(
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
            )

            for t in tasks:
                sig_flag = "✓ **YES**" if t.is_significant else "✗ NO"
                verdict_badge = f"**{t.decision}**" if t.decision == "WIN" else (t.decision if t.decision == "TIE" else f"*{t.decision}*")
                lines.append(
                    f"| `{t.task_id}` | {t.n_seeds} | {t.mean_cand_cost:.4f} ± {t.sem_cand_cost:.4f} | "
                    f"{t.mean_base_cost:.4f} ± {t.sem_base_cost:.4f} | {t.r_rb:>+0.2f} | {t.p_raw:.4f} | "
                    f"{t.p_holm:.4f} | {sig_flag} | {verdict_badge} |"
                )
            lines.append("")

        with open(md_file, "w") as f:
            f.write("\n".join(lines) + "\n")

        return md_file

    def export_latex(self, results: Dict[str, Tuple[List[TaskWilcoxonResult], AggregateSummary]]) -> str:
        tex_file = os.path.join(self.output_dir, "1v1_wilcoxon_table.tex")
        lines = [
            "% Auto-generated 1v1 Wilcoxon Statistical Comparison Table",
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\caption{1v1 Head-to-Head Paired Wilcoxon Signed-Rank Test vs. \\texttt{SMAC3\\_HPOFacade\\_ei} across seeds ($\\alpha=0.05$).}",
            "\\label{tab:1v1_wilcoxon_results}",
            "\\begin{tabular}{lrrrrrrc}",
            "\\toprule",
            "\\textbf{Task Benchmark} & \\textbf{Cand. Mean $\\pm$ SEM} & \\textbf{Base. Mean $\\pm$ SEM} & $\\mathbf{\\Delta}$ (\\%) & $\\mathbf{r_{\\text{rb}}}$ & $\\mathbf{p_{\\text{raw}}}$ & $\\mathbf{p_{\\text{Holm}}}$ & \\textbf{Verdict} \\\\",
            "\\midrule"
        ]

        for cand_id, (tasks, agg) in results.items():
            cand_clean = cand_id.replace("_", "\\_")
            lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{Approach:}} \\texttt{{{cand_clean}}}}} \\\\")
            lines.append("\\midrule")
            for t in tasks:
                task_clean = t.task_id.replace("subset_yahpo_", "").replace("subset_hpobench_", "").replace("_", "\\_")
                verdict_tex = f"\\textbf{{{t.decision}}}" if t.decision == "WIN" else t.decision
                lines.append(
                    f"{task_clean} & {t.mean_cand_cost:.4f} $\\pm$ {t.sem_cand_cost:.4f} & "
                    f"{t.mean_base_cost:.4f} $\\pm$ {t.sem_base_cost:.4f} & {t.rel_improvement_pct:>+0.1f}\\% & "
                    f"{t.r_rb:>+0.2f} & {t.p_raw:.4f} & {t.p_holm:.4f} & {verdict_tex} \\\\"
                )
            lines.append("\\midrule")
            lines.append(
                f"\\textbf{{Macro Aggregate}} & \\textbf{{{agg.mean_cost_cand:.4f}}} & \\textbf{{{agg.mean_cost_base:.4f}}} & "
                f"\\textbf{{{agg.global_rel_improvement_pct:>+0.1f}\\%}} & \\textbf{{{agg.macro_r_rb:>+0.2f}}} & "
                f"\\textbf{{{agg.macro_p:.4f}}} & --- & \\textbf{{{agg.wins}W / {agg.ties}T / {agg.losses}L}} \\\\"
            )
            lines.append("\\midrule")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}"
        ])

        with open(tex_file, "w") as f:
            f.write("\n".join(lines) + "\n")

        return tex_file

    def export_csv(self, results: Dict[str, Tuple[List[TaskWilcoxonResult], AggregateSummary]]) -> str:
        csv_file = os.path.join(self.output_dir, "1v1_wilcoxon_summary.csv")
        rows = []
        for cand_id, (tasks, agg) in results.items():
            for t in tasks:
                rows.append({
                    "candidate_id": cand_id,
                    "baseline_id": agg.baseline_id,
                    "task_id": t.task_id,
                    "n_seeds": t.n_seeds,
                    "mean_cand_cost": t.mean_cand_cost,
                    "sem_cand_cost": t.sem_cand_cost,
                    "mean_base_cost": t.mean_base_cost,
                    "sem_base_cost": t.sem_base_cost,
                    "mean_diff": t.mean_diff,
                    "rel_improvement_pct": t.rel_improvement_pct,
                    "w_plus": t.w_plus,
                    "w_minus": t.w_minus,
                    "wilcoxon_w": t.wilcoxon_w,
                    "p_raw": t.p_raw,
                    "p_holm": t.p_holm,
                    "p_bh": t.p_bh,
                    "r_rb": t.r_rb,
                    "cliffs_delta": t.cliffs_delta,
                    "cohens_d": t.cohens_d,
                    "is_significant": t.is_significant,
                    "decision": t.decision
                })
        pd.DataFrame(rows).to_csv(csv_file, index=False)
        return csv_file


def run_full_1v1_analysis(
    parquet_path: str = "results/sweep_1v1_analysis/logs.parquet",
    baseline_id: str = "SMAC3_HPOFacade_ei",
    candidates: Optional[List[str]] = None,
    alpha: float = 0.05,
    output_dir: str = "results/sweep_1v1_analysis/report_1v1_sweeps"
) -> Dict[str, Tuple[List[TaskWilcoxonResult], AggregateSummary]]:
    """Loads parquet logs and computes 1v1 Wilcoxon comparisons for all approaches against baseline."""
    print(f"\n[1/4] Loading CARP-S logs from: {parquet_path}...")
    df = DataLoader.load_parquet(parquet_path)
    print(f"      Loaded {len(df):,} total evaluations across {df['task_id'].nunique()} tasks and {df['seed'].nunique()} seeds.")

    print("\n[2/4] Extracting terminal incumbent costs...")
    df_terminal = DataLoader.extract_terminal_costs(df)

    # Discover candidate optimizers (excluding baseline)
    all_optimizers = df_terminal["optimizer_id"].unique().tolist()
    if candidates is None:
        candidates = [opt for opt in all_optimizers if opt != baseline_id]
    
    if not candidates:
        raise ValueError(f"No candidate optimizers found distinct from baseline '{baseline_id}' in {all_optimizers}")

    print(f"      Baseline:   {baseline_id}")
    print(f"      Candidates: {candidates}")

    engine = StatisticalAnalysisEngine(alpha=alpha)
    results = {}

    print(f"\n[3/4] Computing paired Wilcoxon signed-rank tests (alpha = {alpha})...")
    for cand in candidates:
        paired_dict = PairingEngine.pair_seeds(df_terminal, cand, baseline_id)
        task_results, agg = engine.analyze_1v1(paired_dict, cand, baseline_id)
        results[cand] = (task_results, agg)

    print("\n[4/4] Generating terminal, Markdown, LaTeX, and CSV reports...")
    exporter = ReportExporter(output_dir=output_dir)
    exporter.print_terminal_report(results)
    
    md_path = exporter.export_markdown(results)
    tex_path = exporter.export_latex(results)
    csv_path = exporter.export_csv(results)
    
    print(f"\n[✓] Reports successfully saved to: {output_dir}/")
    print(f"    - Markdown: {md_path}")
    print(f"    - LaTeX:    {tex_path}")
    print(f"    - CSV:      {csv_path}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1v1 Wilcoxon Statistical Analysis Suite for CARP-S")
    parser.add_argument("--parquet", type=str, default="results/sweep_1v1_analysis/logs.parquet", help="Path to logs.parquet")
    parser.add_argument("--baseline", type=str, default="SMAC3_HPOFacade_ei", help="Optimizer ID for baseline")
    parser.add_argument("--candidates", nargs="*", default=None, help="List of candidate optimizer IDs")
    parser.add_argument("--alpha", type=float, default=0.05, help="Statistical significance threshold alpha")
    parser.add_argument("--output-dir", type=str, default="results/sweep_1v1_analysis/report_1v1_sweeps", help="Report output directory")
    args = parser.parse_args()

    run_full_1v1_analysis(
        parquet_path=args.parquet,
        baseline_id=args.baseline,
        candidates=args.candidates,
        alpha=args.alpha,
        output_dir=args.output_dir
    )
