import argparse

import numpy as np
import pandas as pd

from document_check import TermChecker
from functions import load_glossary

DOC_PATH = "data/sample_doc.txt"
LABELS_PATH = "data/test_labels.csv"


def norm(term):
    return term.strip().lower().replace("ё", "е")


def load_labels(text, path=LABELS_PATH):
    labels = []
    for _, r in pd.read_csv(path, dtype=str).iterrows():
        start = text.find(r["phrase"])
        if start < 0:
            raise ValueError(f"Фразы «{r['phrase']}» из {path} нет в документе")
        labels.append({"phrase": r["phrase"], "kind": r["kind"],
                       "expected": {norm(t) for t in r["expected"].split(";")},
                       "start": start, "end": start + len(r["phrase"])})
    return labels


def score(findings, labels):
    def hits(f, lab):
        overlap = f["start"] < lab["end"] and lab["start"] < f["end"]
        return overlap and norm(f["suggestion"]) in lab["expected"]

    correct = [f for f in findings if any(hits(f, lab) for lab in labels)]
    found = [lab for lab in labels if any(hits(f, lab) for f in findings)]
    precision = len(correct) / len(findings) if findings else 0.0
    recall = len(found) / len(labels)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "findings": len(findings), "correct": len(correct),
        "precision": precision, "recall": recall, "f1": f1,
        "recall_variants": sum(lab["kind"] == "вариант" for lab in found),
        "recall_semantic": sum(lab["kind"] == "смысл" for lab in found),
        "missed": [lab for lab in labels if lab not in found],
        "extra": [f for f in findings if f not in correct],
    }


def main():
    parser = argparse.ArgumentParser(description="Оценка качества и подбор порога")
    parser.add_argument("--doc", default=DOC_PATH)
    parser.add_argument("--labels", default=LABELS_PATH)
    parser.add_argument("--threshold", type=float, help="проверить только этот порог")
    parser.add_argument("--llm", action="store_true",
                        help="добавить уровень 3: проверку языковой моделью (нужна Ollama)")
    parser.add_argument("--model", help="модель Ollama для --llm, например qwen2.5:3b")
    args = parser.parse_args()

    with open(args.doc, encoding="utf-8") as f:
        text = f.read()
    labels = load_labels(text, args.labels)
    if args.llm:
        from llm_check import LLMChecker
        from functions import LLM_MODEL
        checker = LLMChecker(load_glossary(), model=args.model or LLM_MODEL)
    else:
        checker = TermChecker(load_glossary())

    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = np.round(np.arange(0.50, 0.951, 0.05), 2)
    rows = []
    for threshold in thresholds:
        s = score(checker.check(text, threshold), labels)
        rows.append({"threshold": threshold, **s})
    table = pd.DataFrame(rows)

    n_var = sum(lab["kind"] == "вариант" for lab in labels)
    n_sem = len(labels) - n_var
    print(f"Размеченных ошибок: {len(labels)} (по вариантам {n_var}, по смыслу {n_sem})\n")
    print("порог  замечаний  верных  precision  recall    F1   найдено: варианты / смысл")
    for _, r in table.iterrows():
        print(f"{r['threshold']:5.2f}  {r['findings']:9d}  {r['correct']:6d}  {r['precision']:9.2f}"
              f"  {r['recall']:6.2f}  {r['f1']:4.2f}   {r['recall_variants']}/{n_var} / "
              f"{r['recall_semantic']}/{n_sem}")

    best = table.sort_values(["f1", "threshold"], ascending=False).iloc[0]
    if args.threshold is None:
        print(f"\nЛучший порог по F1: {best['threshold']:.2f} (F1 = {best['f1']:.2f})")
    print("\nПропущенные ошибки:")
    for lab in best["missed"] or []:
        print(f"  «{lab['phrase']}» (нужно: {', '.join(sorted(lab['expected']))})")
    print("\nЛишние или неверные замечания:")
    for f in best["extra"] or []:
        conf = "" if f["score"] is None else f", {f['score']}"
        print(f"  «{f['phrase']}» → «{f['suggestion']}» [{f['method']}{conf}]")

    columns = ["threshold", "findings", "correct", "precision", "recall", "f1",
               "recall_variants", "recall_semantic"]
    if args.threshold is None:
        out = "reports/threshold_metrics_llm.csv" if args.llm else "reports/threshold_metrics.csv"
        table[columns].round(3).to_csv(out, index=False)
        print(f"\nТаблица сохранена в {out}")

if __name__ == "__main__":
    main()