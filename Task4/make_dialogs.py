"""Прогон тестовых вопросов и запись Task4/dialogs.md. Запуск из корня: .venv/bin/python Task4/make_dialogs.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from rag import LLM_MODEL, RagBot  # noqa: E402

# (вопрос, есть ли ответ в базе)
QUESTIONS = [
    ("Кто убил Корвина Арсеновича?", True),
    ("К какому наказанию приговорили Радия на суде?", True),
    ("Кто такой Верховин?", True),
    ("Кто такая Анфисушка?", True),
    ("Кого ударил Радий медным пестиком, убегая из сада?", True),
    ("Что произошло с телом Талимы после смерти?", True),
    ("Кто вырастил Радия, Ларена и Лиошу, когда отец их забыл?", True),
    ("Какая погода в Москве?", False),
    ("Сколько денег лежит у Гордеева в банке?", False),
]


def main():
    bot = RagBot()
    out = ["# Примеры диалогов\n",
           f"Модель `{LLM_MODEL}`, индекс `Task3/chroma_db/`, top-4 чанков, порог отказа по расстоянию 0.5.\n"]
    for i, (q, in_base) in enumerate(QUESTIONS):
        t = time.time()
        r = bot.answer(q, pre_prompt=False, filter_chunks=False, sanitize=False)  # конфигурация Задания 4: без защиты
        dt = time.time() - t
        if in_base and r.refused:
            kind = "ложный отказ (ответ в базе есть)"
        elif in_base:
            kind = "успешный ответ"
        elif r.raw:
            kind = "отказ: «Я не знаю» от модели"
        else:
            kind = "отказ до LLM (расстояние выше порога)"
        src = ", ".join(f"`{os.path.basename(c.source)}` ({c.distance:.2f})" for c in r.chunks)
        out.append(f"\n## {i + 1}. {kind}\n\n**Вы:** {q}\n\n**Бот:**\n\n```\n{r.answer}\n```\n\n"
                   f"Найденные чанки: {src}. Время: {dt:.1f} с.\n")
        print(i + 1, kind, f"{dt:.1f}s")
        print(r.answer, "\n")
    with open("Task4/dialogs.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
