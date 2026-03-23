"""
End-of-Day Reporter
Generates a rich HTML report of the day's trading activity,
saves it to the reports/ directory, and pushes to GitHub.
The report is served via GitHub Pages or Cloudflare Pages — free.
"""

import os
import json
import logging
import subprocess
from datetime import date, datetime
from typing import Dict, List, Optional
from agents.risk_manager import RiskManagerAgent

logger = logging.getLogger(__name__)


class EODReporter:

    def __init__(self, risk_mgr: RiskManagerAgent = None):
        self.risk_mgr = risk_mgr or RiskManagerAgent()
        self.report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        os.makedirs(self.report_dir, exist_ok=True)

    def generate(self) -> str:
        """Generate today's EOD report and return the file path."""
        summary = self.risk_mgr.get_daily_summary()
        html    = self._render_html(summary)
        json_data = self._render_json(summary)

        today    = date.today().isoformat()
        html_path = os.path.join(self.report_dir, f"report_{today}.html")
        json_path = os.path.join(self.report_dir, f"report_{today}.json")
        index_path = os.path.join(self.report_dir, "index.html")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_data)

        # Update index.html to link to this report
        self._update_index(summary, today)

        logger.info(f"EOD report generated: {html_path}")
        self._git_commit_and_push(today)

        return html_path

    def _render_html(self, summary: Dict) -> str:
        pnl = summary.get("total_pnl", 0)
        pnl_color = "#16a34a" if pnl >= 0 else "#dc2626"
        pnl_sign  = "+" if pnl >= 0 else ""
        trades    = summary.get("trades", [])

        trade_rows = ""
        for t in trades:
            t_pnl = t.get("pnl", 0)
            t_color = "#16a34a" if t_pnl >= 0 else "#dc2626"
            trade_rows += f"""
            <tr>
                <td>{t.get('symbol','')}</td>
                <td><span class="badge badge-short">SHORT</span></td>
                <td>₹{t.get('entry', 0):.2f}</td>
                <td>₹{t.get('exit', 0) or '—'}</td>
                <td>{t.get('qty', 0)}</td>
                <td style="color:{t_color};font-weight:600">
                    {'+' if t_pnl>=0 else ''}₹{t_pnl:.0f}
                </td>
                <td><span class="badge badge-{t.get('status','').lower()}">{t.get('status','')}</span></td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tej — {summary.get('date','')}</title>
<style>
  :root {{
    --bg: #f8fafc; --surface: #fff; --border: #e2e8f0;
    --text: #1e293b; --muted: #64748b;
    --green: #16a34a; --red: #dc2626; --blue: #2563eb;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: var(--bg); color: var(--text); padding: 2rem; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  header {{ margin-bottom: 2rem; }}
  header h1 {{ font-size: 1.5rem; font-weight: 700; }}
  header p  {{ color: var(--muted); margin-top: .25rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
             gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 12px; padding: 1.25rem; }}
  .card-label {{ font-size: .75rem; color: var(--muted); text-transform: uppercase;
                 letter-spacing: .05em; margin-bottom: .5rem; }}
  .card-value {{ font-size: 1.5rem; font-weight: 700; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface);
           border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
  th {{ background: #f1f5f9; font-size: .75rem; text-transform: uppercase;
        letter-spacing: .05em; color: var(--muted); padding: .75rem 1rem; text-align: left; }}
  td {{ padding: .75rem 1rem; border-top: 1px solid var(--border); font-size: .9rem; }}
  .badge {{ padding: .2rem .6rem; border-radius: 9999px; font-size: .7rem; font-weight: 600; }}
  .badge-short  {{ background: #fef2f2; color: #dc2626; }}
  .badge-open   {{ background: #eff6ff; color: #2563eb; }}
  .badge-closed {{ background: #f0fdf4; color: #16a34a; }}
  footer {{ margin-top: 2rem; color: var(--muted); font-size: .8rem; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Tej — Daily Report</h1>
    <p>{summary.get('date','')} · Generated {datetime.now().strftime('%H:%M IST')}</p>
  </header>

  <div class="cards">
    <div class="card">
      <div class="card-label">Total P&L</div>
      <div class="card-value" style="color:{pnl_color}">{pnl_sign}₹{pnl:,.0f}</div>
    </div>
    <div class="card">
      <div class="card-label">Total Trades</div>
      <div class="card-value">{summary.get('win_count',0) + summary.get('loss_count',0)}</div>
    </div>
    <div class="card">
      <div class="card-label">Win Rate</div>
      <div class="card-value">{summary.get('win_rate',0):.0f}%</div>
    </div>
    <div class="card">
      <div class="card-label">Winners</div>
      <div class="card-value green">{summary.get('win_count',0)}</div>
    </div>
    <div class="card">
      <div class="card-label">Losers</div>
      <div class="card-value red">{summary.get('loss_count',0)}</div>
    </div>
    <div class="card">
      <div class="card-label">Best Trade</div>
      <div class="card-value">{summary.get('best_trade','—')}</div>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Symbol</th><th>Direction</th><th>Entry</th>
        <th>Exit</th><th>Qty</th><th>P&L</th><th>Status</th>
      </tr>
    </thead>
    <tbody>
      {trade_rows if trade_rows else '<tr><td colspan="7" style="text-align:center;color:#94a3b8">No trades today</td></tr>'}
    </tbody>
  </table>

  <footer>Tej Autonomous Agent · Paper trade mode active</footer>
</div>
</body>
</html>"""

    def _render_json(self, summary: Dict) -> str:
        return json.dumps(summary, indent=2, default=str)

    def _update_index(self, summary: Dict, today: str):
        """Maintain a running index.html with links to all daily reports."""
        index_path = os.path.join(self.report_dir, "index.html")
        pnl = summary.get("total_pnl", 0)
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_color = "#16a34a" if pnl >= 0 else "#dc2626"

        new_row = f"""<tr>
          <td><a href="report_{today}.html">{today}</a></td>
          <td style="color:{pnl_color};font-weight:600">{pnl_sign}₹{pnl:,.0f}</td>
          <td>{summary.get('win_count',0)}W / {summary.get('loss_count',0)}L</td>
          <td>{summary.get('win_rate',0):.0f}%</td>
        </tr>"""

        # Read existing rows
        existing = ""
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                content = f.read()
            # Extract existing rows
            if "<tbody>" in content:
                existing = content.split("<tbody>")[1].split("</tbody>")[0]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tej — Reports</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 700px;
          margin: 2rem auto; padding: 0 1rem; color: #1e293b; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 1rem; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: .65rem 1rem; border-bottom: 1px solid #e2e8f0; text-align: left; }}
  th {{ background: #f8fafc; font-size: .8rem; text-transform: uppercase; color: #64748b; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>Tej — Trade Reports</h1>
<table>
  <thead><tr><th>Date</th><th>P&L</th><th>Trades</th><th>Win Rate</th></tr></thead>
  <tbody>
    {new_row}
    {existing}
  </tbody>
</table>
</body>
</html>"""
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _git_commit_and_push(self, today: str):
        """Commit the daily report to GitHub (which triggers Cloudflare Pages deploy)."""
        try:
            repo_root = os.path.join(os.path.dirname(__file__), "..")
            cmds = [
                ["git", "-C", repo_root, "add", "reports/"],
                ["git", "-C", repo_root, "commit", "-m", f"chore: EOD report {today}"],
                ["git", "-C", repo_root, "push", "origin", "main"],
            ]
            for cmd in cmds:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0 and "nothing to commit" not in result.stdout:
                    logger.warning(f"Git command warning: {result.stderr[:100]}")

            logger.info("EOD report pushed to GitHub")
        except Exception as e:
            logger.warning(f"Git push failed (non-critical): {e}")
