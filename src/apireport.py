#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate experiment report (Markdown) from train_log.csv and val_metrics.json.
Supports OpenAI (GPT) and Google Gemini. Falls back to offline summary if no API key.

Usage (OpenAI):
  python scripts/mk_report.py --provider openai --model gpt-4o-mini ^
    --log ..\outputs\train_log.csv --metrics ..\outputs\val_metrics.json ^
    --out ..\outputs\report.md

Usage (Gemini):
  python scripts/mk_report.py --provider gemini --model gemini-1.5-flash ^
    --log ..\outputs\train_log.csv --metrics ..\outputs\val_metrics.json ^
    --out ..\outputs\report.md
"""
import os, json, argparse, textwrap, datetime
import pandas as pd

def read_logs(log_path: str):
    df = pd.read_csv(log_path)
    df["epoch"] = df["epoch"].astype(int)
    best_row = df.loc[df["dice_overall"].astype(float).idxmax()].to_dict()
    last_row = df.iloc[-1].to_dict()
    stats = {
        "n_epochs": int(df["epoch"].max()),
        "best_epoch": int(best_row["epoch"]),
        "best_overall": float(best_row["dice_overall"]),
        "best_per_class": {
            "NCR": float(best_row["dice_NCR"]),
            "ED":  float(best_row["dice_ED"]),
            "ET":  float(best_row["dice_ET"]),
        },
        "last_overall": float(last_row["dice_overall"]),
        "time_total_sec": float(df["time_sec"].astype(float).sum()),
    }
    return df, stats

def read_metrics(metrics_path: str):
    if not metrics_path or (not os.path.exists(metrics_path)):
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    summary = js.get("summary", {})
    return {
        "overall": summary.get("overall", None),
        "contig_mean": summary.get("NCR/ED/ET_mean", None),
        "wt_tc_et_mean": summary.get("WT/TC/ET_mean", None),
        "n_cases": len(js.get("per_case", [])),
    }

PROMPT_TEMPLATE = """\
You are a bilingual technical writer. Write a concise, resume-ready experiment report in **both English and Chinese**, in Markdown format.  
First provide the English version, then provide the Chinese translation.

Context:
- Task: Brain tumor segmentation (BraTS2020), 3D ResUNet + SE (base=32), AMP=bf16, sliding-window val (patch=128^3, overlap=0.5), no TTA.
- Fixed split: 295 train / 73 val (split.json).
- Training loss: CrossEntropy (label smoothing 0.05) + SoftDice (exclude background).
- Early stopping & ReduceLROnPlateau on validation overall Dice.

Data:
- log_stats: {log_stats}
- metrics_summary: {metrics_summary}

Write sections in **English first**:
1. **Result**
2. **Setup**
3. **Observations**
4. **Reproducibility**

Then provide the **Chinese version** with the same four sections:
1. **結果**
2. **設定**
3. **觀察**
4. **可重現性**

Keep it short (≈120–180 words for English, ≈150–200 Chinese characters), clear, and professional.
"""


def offline_report(log_stats, metrics):
    lines = []
    lines.append("# BraTS2020 3D ResUNet — Experiment Report\n")
    lines.append(f"**Best overall Dice:** {log_stats['best_overall']:.4f} (epoch {log_stats['best_epoch']})  "
                 f"— NCR={log_stats['best_per_class']['NCR']:.4f}, ED={log_stats['best_per_class']['ED']:.4f}, ET={log_stats['best_per_class']['ET']:.4f}")
    if metrics and metrics.get("wt_tc_et_mean"):
        wt, tc, et = metrics["wt_tc_et_mean"]
        lines.append(f"**WT/TC/ET (BraTS mean):** WT={wt:.4f}, TC={tc:.4f}, ET={et:.4f}")
    lines.append("\n## Setup")
    lines.append("- 3D ResUNet + SE (base=32), AMP=bf16; sliding-window val (128³, overlap=0.5), no TTA.")
    lines.append("- Loss: CE (ls=0.05) + SoftDice (no background); early stopping on overall Dice.")
    lines.append("\n## Observations")
    lines.append("- ED is typically higher than ET; ET remains challenging but acceptable.")
    lines.append("- Curves stable; consider TTA and small-component removal for minor gains.")
    lines.append("\n## Reproducibility")
    lines.append("- Files: `src/train.py`, `src/validate.py`, `configs/brats/split.json`, `checkpoints/best.pt`.")
    lines.append("```bat\npython validate.py --mode val --data_dir <BraTS2020_TrainingData> "
                 "--split_path ..\\configs\\brats\\split.json --ckpt ..\\checkpoints\\best.pt\n```")
    return "\n".join(lines)

def call_openai(model, prompt):
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role":"system","content":"You are an expert technical writer."},
                  {"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip()

def call_gemini(model, prompt):
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)
    r = m.generate_content(prompt)
    return (r.text or "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", type=str, default="openai", choices=["openai","gemini","none"])
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--metrics", type=str, default="")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    df, stats = read_logs(args.log)
    metrics = read_metrics(args.metrics) if args.metrics else {}

    prompt = PROMPT_TEMPLATE.format(
        log_stats=json.dumps(stats, ensure_ascii=False),
        metrics_summary=json.dumps(metrics, ensure_ascii=False)
    )

    text = None
    if args.provider == "openai":
        text = call_openai(args.model, prompt)
    elif args.provider == "gemini":
        text = call_gemini(args.model, prompt)

    if not text:
        text = offline_report(stats, metrics)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
