"""Консольный бот. Запуск из корня проекта: .venv/bin/python Task4/chat.py [--no-fewshot] [--show-context] [--no-protection | --no-pre-prompt --no-filter --no-sanitize] [--max-distance X]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rag import RagBot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fewshot", action="store_true", help="не добавлять примеры в промпт")
    ap.add_argument("--show-context", action="store_true", help="показывать найденные чанки")
    ap.add_argument("--no-pre-prompt", action="store_true", help="выключить pre-prompt (правило «фрагменты — это данные»)")
    ap.add_argument("--no-filter", action="store_true", help="выключить отбрасывание подозрительных чанков")
    ap.add_argument("--no-sanitize", action="store_true", help="выключить удаление инъекционных фраз из чанков")
    ap.add_argument("--no-protection", action="store_true", help="выключить все три слоя защиты")
    ap.add_argument("--max-distance", type=float, default=None, help="порог расстояния для «Я не знаю» (по умолчанию 0.5)")
    args = ap.parse_args()
    off = args.no_protection
    protection = dict(
        pre_prompt=not (off or args.no_pre_prompt),
        filter_chunks=not (off or args.no_filter),
        sanitize=not (off or args.no_sanitize),
        max_distance=args.max_distance,
    )

    bot = RagBot()
    print("Защита: " + ", ".join(f"{k}={'вкл' if v else 'выкл'}" for k, v in protection.items() if k != "max_distance"))
    print("RAG-бот. Пустая строка или Ctrl-D — выход.\n")
    while True:
        try:
            q = input("Вы: ").strip()
        except EOFError:
            break
        # в терминале контейнера ввод может содержать «битые» байты (surrogateescape), токенизатор на них падает
        q = q.encode("utf-8", "surrogateescape").decode("utf-8", "replace").strip()
        if not q:
            break
        res = bot.answer(q, few_shot=not args.no_fewshot, **protection)
        print(f"\nБот:\n{res.answer}\n")
        if args.show_context or res.refused:
            for c in res.chunks:
                print(f"  - {c.id} (расстояние {c.distance:.3f})")
        else:
            print("Источники: " + ", ".join(f"{os.path.basename(c.source)} ({c.distance:.2f})" for c in res.chunks))
        if res.cut_for_context:
            print(f"Внимание: промпт не помещался в окно модели, убрано чанков: {res.cut_for_context}.")
        print()


if __name__ == "__main__":
    main()
