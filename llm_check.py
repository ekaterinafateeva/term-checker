import argparse
import json
import os

import pandas as pd

from document_check import DEFAULT_THRESHOLD, TermChecker, contains_error
from functions import LLM_MODEL, ask_llm, closest_articles, load_glossary

TOP_K = 5
MAX_PHRASE_WORDS = 6  # длиннее — модель процитировала пол-предложения вместо ошибки
# Формат ответа: модель обязана вернуть список ошибок, каждая — цитата и термин
SCHEMA = {
    "type": "object",
    "properties": {"errors": {"type": "array", "items": {
        "type": "object",
        "properties": {"phrase": {"type": "string"}, "term": {"type": "string"}},
        "required": ["phrase", "term"],
    }}},
    "required": ["errors"],
}


def norm(term):
    return term.strip().strip("«»\"").lower().replace("ё", "е")


def build_prompt(sentence, articles):
    # Запрос модели: статьи справочника, что считать ошибкой, два примера и само предложение
    lines = []
    for r in articles.itertuples():
        line = f"- «{r.term}»: {r.definition or 'определения нет'}"
        if r.variants:
            line += f" Неутверждённые варианты: {r.variants.replace(';', ',')}."
        lines.append(line)
    return (
        "Ты проверяешь корпоративный документ на соответствие справочнику терминов.\n\n"
        "Статьи справочника:\n" + "\n".join(lines) + "\n\n"
        "Ошибка — это слово или словосочетание, которое означает ровно то же понятие, что и термин "
        "из списка, но записано иначе: синонимом, жаргоном, сокращением или описанием своими словами. "
        "Такую фразу можно заменить термином, и смысл предложения не изменится.\n"
        "Не ошибка:\n"
        "- утверждённый термин в любом падеже;\n"
        "- слова на ту же тему, которые не называют само понятие: часть понятия, пример, действие, "
        "подразделение, документ.\n\n"
        "Примеры:\n"
        "Предложение: «Кредиторка выросла на 5%.» Ответ: {\"errors\": [{\"phrase\": \"Кредиторка\", "
        "\"term\": \"Кредиторская задолженность\"}]}\n"
        "Предложение: «Мы купили новые здания и станки.» Ответ: {\"errors\": []} — здания и станки "
        "лишь примеры основных средств, а не другое название понятия.\n\n"
        f"Предложение: «{sentence}»\n"
        "Для каждой ошибки укажи phrase — точную короткую цитату из предложения — и term — "
        "утверждённый термин из списка. Если ошибок нет, верни пустой список errors."
    )


class LLMChecker(TermChecker):
    # Уровни 1–2 как в TermChecker и уровень 3: языковая модель проверяет каждое предложение

    def __init__(self, glossary, threshold=DEFAULT_THRESHOLD, model=LLM_MODEL):
        super().__init__(glossary, threshold)
        self.glossary = glossary
        self.model = model
        self.all_terms = list(glossary["term"])
        self._answers = {}  # ответы модели: при повторной проверке не спрашиваем заново

    def ask(self, sentence, articles):
        # Спросить модель и достать из ответа пары (цитата, термин)
        prompt = build_prompt(sentence, articles)
        if prompt not in self._answers:
            self._answers[prompt] = ask_llm(prompt, self.model, SCHEMA)
        try:
            errors = json.loads(self._answers[prompt])["errors"]
        except (ValueError, KeyError, TypeError):
            return []
        return [(e["phrase"].strip(" «»\".,"), e["term"]) for e in errors
                if isinstance(e, dict) and e.get("phrase") and e.get("term")]

    def check_sentence(self, sentence, threshold):
        found = super().check_sentence(sentence, threshold)
        articles = closest_articles(sentence, self.glossary, TOP_K)
        allowed = {norm(t): t for t in articles["term"]}

        for phrase, term in self.ask(sentence, articles):
            start = sentence.lower().find(phrase.lower())
            term = allowed.get(norm(term))
            if start < 0 or term is None or len(phrase.split()) > MAX_PHRASE_WORDS:
                continue  # цитаты нет в предложении или термина нет среди статей: модель выдумала
            end = start + len(phrase)
            if any(start < f["end"] and f["start"] < end for f in found):
                continue  # это место уже отмечено правилом или эмбеддингами
            if any(not contains_error(phrase, t) for t in self.all_terms):
                continue  # во фразе утверждённый термин или его часть: это не ошибка
            found.append({"phrase": sentence[start:end], "suggestion": term,
                          "method": "языковая модель", "score": None,
                          "start": start, "end": end})
        return sorted(found, key=lambda f: f["start"])


def main():
    parser = argparse.ArgumentParser(description="Проверка документа с языковой моделью")
    parser.add_argument("path", help="текстовый файл в кодировке UTF-8")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--model", default=LLM_MODEL)
    args = parser.parse_args()

    with open(args.path, encoding="utf-8") as f:
        text = f.read()
    findings = LLMChecker(load_glossary(), args.threshold, args.model).check(text)

    print(f"Замечаний: {len(findings)}\n")
    for f in findings:
        score = "" if f["score"] is None else f", {f['score']}"
        print(f"  «{f['phrase']}» → используйте «{f['suggestion']}» [{f['method']}{score}]")

    os.makedirs("reports", exist_ok=True)
    columns = ["sentence", "phrase", "suggestion", "method", "score"]
    pd.DataFrame(findings, columns=columns).to_csv("reports/llm_check.csv", index=False)
    print("\nТаблица сохранена в reports/llm_check.csv")


if __name__ == "__main__":
    main()