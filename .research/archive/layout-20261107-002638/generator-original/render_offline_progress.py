"""Render the first offline-RL batch status without conflating preparation and training."""

import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".runtime/experiments/e4-offline-rl"
OUT = ROOT / ".research"


def read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    plan = read(RUN / "plan.json", {})
    pool = read(RUN / "pool-manifest.json", {})
    pipeline = read(OUT / "offline-rl-pipeline-status.json", {})
    collection = read(OUT / "offline-batch-status.json", {})
    phase = pipeline.get("phase", "preparing")
    labels = {
        "waiting for fixed pool and official backbones": "等待官方价值模型下载完成",
        "waiting for GPU lock": "等待 GPU 资源",
        "official Gemma3/SigLIP2 critic smoke test": "官方价值模型前向 / 反向验收",
        "training official value model and labeling advantages": "价值学习与优势标注",
        "pilot policy training and held-out evaluation": "策略训练与配对评估",
        "complete": "首轮全部完成", "error": "需检查错误日志",
    }
    rows = []
    for task in collection.get("tasks", []):
        rows.append(f'<tr><td>{html.escape(task["task"])}</td><td>{task["episodes"]}/40</td><td>{task["successes"]}</td><td>{task["failures"]}</td></tr>')
    jobs = []
    for job in sorted((RUN / "jobs").glob("*")) if (RUN / "jobs").exists() else []:
        progress = read(job / "progress.json", {})
        contract = read(job / "contract.json", {})
        result = read(job / "result.json", {})
        evaluation = read(OUT / "replay-evaluation-status.json", {})
        fallback = "评估中" if job.name == "mt5-success-replay" and evaluation.get("phase") == "evaluating" else ("训练完成，待评估" if (job / "FINISHED").exists() else "训练中")
        score = "；".join(f'{r["task"]}: {r["successes"]}/20（原始 {r["baseline_successes_same_seeds"]}/20）' for r in result.get("results", []))
        jobs.append(f'<tr><td>{html.escape(job.name)}</td><td>{progress.get("step", 0)}/{contract.get("steps", "待启动")}</td><td>{html.escape(result.get("status", fallback))}</td><td>{html.escape(score or "尚无评估结果")}</td></tr>')
    gemma = (ROOT / "models/gemma-3-270m/DOWNLOAD_COMPLETE").exists()
    siglip = (ROOT / "models/siglip2-so400m-patch14-224/DOWNLOAD_COMPLETE").exists()
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Offline RL 首轮批量实验</title><style>body{{max-width:1100px;margin:35px auto;padding:0 22px;font:17px/1.8 system-ui;color:#172b3a;background:#f5f7f9}}section{{background:white;padding:24px;border:1px solid #dce5e9;border-radius:9px;margin:20px 0}}h1{{font-size:29px}}h2{{font-size:22px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce5e9}}a{{color:#087e80}}.stage{{font-size:22px;color:#087e80;font-weight:bold}}</style></head><body><h1>Offline RL 首轮批量实验</h1><p><a href="e4-analysis.html">查看首轮结果分析：原始 47%，单任务 RL 29%，多任务 RL 29%，成功重放 25%</a></p><p>更新：{now}（北京时间）</p><section><h2>当前阶段</h2><p class="stage">{html.escape(labels.get(phase, phase))}</p><p>RL 策略优化已启动：{"是" if pipeline.get("policy_training_started", False) else "否"}。准备数据、价值训练和策略训练分开记录。</p><p>{html.escape(str(pipeline.get("error", "")))}</p></section><section><h2>算法与固定预算</h2><p>复用 RLinf RECAP：官方 SigLIP2 + Gemma3 价值模型、32 步优势、top 30% 优势条件化 flow matching 更新。首轮冻结价值主干和策略 VLM，训练 critic expert 与策略 action expert，属于快速验证设置。</p><p>Gemma3：{"下载已校验" if gemma else "下载中"}；SigLIP2：{"下载已校验" if siglip else "下载中"}。</p><p>冻结经验池：{pool.get("episodes", 0)} 条轨迹，{pool.get("frames", 0)} 个观测 / 动作记录。250 条示范 + 200 条自主轨迹；每任务 32 条自主训练轨迹、8 条价值验证轨迹，示范每任务 45/5 划分。未使用测试轨迹训练。</p><p>首轮 5 个单任务策略 + 1 个五任务共享策略 + 1 个成功经验重放对照。共享同一价值标签以隔离策略共享的影响。单任务 50 次更新，多任务 250 次更新，有效 batch 32；每任务 20 次配对评估。暂不自动扩至 10—50 任务。</p></section><section><h2>自主经验采集</h2><table><thead><tr><th>任务</th><th>完成</th><th>成功</th><th>失败</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section><section><h2>策略作业</h2><table><thead><tr><th>作业</th><th>更新步数</th><th>状态</th><th>配对评估</th></tr></thead><tbody>{"".join(jobs) or '<tr><td colspan="4">尚未启动策略训练；没有 RL 成功率结果。</td></tr>'}</tbody></table></section><section><h2>证据与日志</h2><p><a href="offline-rl-pipeline-status.json">训练队列状态</a> · <a href="offline-batch-status.json">数据采集状态</a> · <a href="../.runtime/experiments/e4-offline-rl/plan.json">实验计划</a> · <a href="../.runtime/experiments/e4-offline-rl/pool-manifest.json">冻结经验清单</a> · <a href="../.runtime/experiments/e4-offline-rl/gemma3-smoke.json">Gemma3 权重与 GPU 验收</a> · <a href="../.runtime/experiments/e4-offline-rl/policy-smoke.json">策略反向验收</a> · <a href="progress.html">总实验记录</a></p></section></body></html>'''
    (OUT / "offline-rl.html").write_text(page)


if __name__ == "__main__":
    main()
