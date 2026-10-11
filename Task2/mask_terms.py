"""Замена ключевых терминов в документах по словарю terms_map.json.

Порядок применения (от более точных правил к менее точным):
  phrases - фразы целиком (слова с контекстом);
  forms   - точные словоформы (слова, где основа меняется или нет склонения);
  stems   - начальная форма -> вымышленная; заменяется основа, окончание сохраняется
            (Зосима, Зосимы, Зосиме -> Талима, Талимы, Талиме...).
files:  имена файлов -> нейтральные имена (чтобы имя файла не выдавало исходный мир).

Использование:
  python3 mask_terms.py <папка_с_исходными> <папка_для_результата> <terms_map.json>
"""
import json
import re
import sys
from pathlib import Path

LETTER = "А-Яа-яЁё"
# допустимые окончания после основы (склонение имён и названий)
ENDINGS = "(?:ыми|ому|ого|ых|ым|ой|ою|ей|ем|ём|ом|ов|ев|ам|ами|ах|ию|ие|ия|ии|а|ы|е|у|и|ю|я|ь|й)?"


def stem(word: str) -> str:
    """Отбрасывает последнюю гласную/й/ь начальной формы, чтобы получить основу."""
    if word.endswith("ий"):
        return word[:-1]
    if word[-1] in "аяйь":
        return word[:-1]
    return word


def tolerant(text: str) -> str:
    """ё и е считаем одной буквой."""
    return re.sub("[её]", "[её]", text, flags=re.IGNORECASE)


def keep_case(original: str, replacement: str) -> str:
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def build_rules(terms: dict):
    rules = []
    left = rf"(?<![{LETTER}])"
    right = rf"(?![{LETTER}])"
    for src in sorted(terms.get("phrases", {}), key=len, reverse=True):
        dst = terms["phrases"][src]
        rules.append((re.compile(left + tolerant(src) + right, re.IGNORECASE),
                      lambda m, d=dst: keep_case(m.group(0), d)))
    for src in sorted(terms.get("forms", {}), key=len, reverse=True):
        dst = terms["forms"][src]
        rules.append((re.compile(left + tolerant(src) + right, re.IGNORECASE),
                      lambda m, d=dst: keep_case(m.group(0), d)))
    for src in sorted(terms.get("stems", {}), key=len, reverse=True):
        dst = terms["stems"][src]
        rules.append((re.compile(left + tolerant(stem(src)) + "(" + ENDINGS + ")" + right),
                      lambda m, d=dst: stem(d) + m.group(1)))
    return rules


def main(src_dir: str, dst_dir: str, map_path: str) -> None:
    terms = json.loads(Path(map_path).read_text(encoding="utf-8"))
    rules = build_rules(terms)
    out = Path(dst_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(src_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for pattern, repl in rules:
            text = pattern.sub(repl, text)
        new_name = terms.get("files", {}).get(path.name, path.name)
        (out / new_name).write_text(text, encoding="utf-8")
        print("masked", path.name, "->", new_name)


if __name__ == "__main__":
    main(*sys.argv[1:4])
