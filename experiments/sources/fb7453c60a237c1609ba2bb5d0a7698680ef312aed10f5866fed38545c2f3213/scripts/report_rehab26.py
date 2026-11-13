"""Собрать понятный отчёт из сохранённых результатов без нового обучения."""

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TITLES = {
    "dummy": "Постоянный ответ",
    "logreg_fixed": "Регрессия, исходные настройки",
    "logreg_tuned": "Регрессия с подбором",
    "svm_tuned": "SVM с подбором",
    "extra_trees_tuned": "Ансамбль деревьев с подбором",
    "inner_selected": "Автоматический выбор по внутренней проверке",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/rehab26_baseline"))
    args = parser.parse_args(argv)
    root = args.report_dir
    from experiment_log import require_open_report
    require_open_report(root)
    results = json.loads((root / "metrics.json").read_text())
    runtime = json.loads((root / "runtime.json").read_text())
    config = json.loads((root / "run_config.json").read_text())
    run_date = datetime.fromisoformat(config["created_utc"]).astimezone(ZoneInfo("Europe/Moscow")).strftime("%d.%m.%Y")
    chosen = next(r for r in results["summaries"] if r["method"] == "inner_selected")
    people = [r for r in results["outer_folds"] if r["method"] == "inner_selected"]
    most_false_feedback = max(people, key=lambda r: r["false_feedback_rate"])
    least_detection = min(people, key=lambda r: r["error_recall"])
    families = dict(Counter(r["candidate"]["family"] for r in people))
    counts = chosen["pooled_repeat_weighted"]["counts_observations"]
    table = "\n".join(
        f"| {TITLES[r['method']]} | {r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f} | {r['false_feedback_rate_mean']:.1%} | {r['error_recall_mean']:.1%} |"
        for r in results["summaries"]
    )
    person_table = "\n".join(
        f"| {r['test_person']} | {r['macro_f1']:.3f} | {r['false_feedback_rate']:.1%} | {r['error_recall']:.1%} |"
        for r in people
    )
    text = f"""# REHAB26: сравнение моделей — {run_date}

## Результат простыми словами

Полностью подготовлены 388 физических приседаний пяти людей: 276 корректных по протоколу набора, 56 со сведением коленей и 56 недостаточно глубоких. Для каждого доступны все три камеры — 1 164 наблюдения. Три камеры не считаются тремя независимыми людьми или повторениями.

Основной автоматический результат: **Macro-F1 {chosen['macro_f1_mean']:.3f} ± {chosen['macro_f1_std']:.3f} по пяти новым людям**. В среднем модель ошибочно критикует {chosen['false_feedback_rate_mean']:.1%} корректных выполнений и замечает {chosen['error_recall_mean']:.1%} ошибочных. Тип ошибки определён верно в {chosen['specific_error_type_recall_mean']:.1%} ошибочных случаев. Ориентиры FPR≤20% и обнаружение ошибок≥80% не достигнуты; это исследовательская модель, её пригодность для автоматической обратной связи не подтверждена.

## Что действительно выполнено

- Скачаны только 15 файлов 3D-поз приседаний: 116 527 874 сжатых байта вместо всего архива; размеры/CRC32 и локальные SHA-256 проверены. Полный MD5 всего ZIP не проверялся, потому что остальные упражнения не скачивались.
- Проверены метки, анатомические индексы, границы массивов, конечность координат/visibility, связанные суставы и состав камер. Низкая visibility не отфильтрована. Нулевые placeholders в полных записях существуют, но не входят в выбранные 1 164 отрезка.
- Получены 35 геометрических признаков целого повторения. Они не используют visibility, порядок фаз или темп. Все признаки рассчитались без исключения выбранных наблюдений.
- Просмотрены галереи пяти людей: по два повторения каждой категории с двух камер; всего 60 наблюдений, по три кадра каждого. Визуализация подтверждает читаемую анатомию и движение; это просмотр скелетов, а не исходных RGB-кадров и не повторная экспертная разметка.
- Обучены исходные методы и проведён ограниченный внутренний подбор по людям. Модели сохранены вместе с предобработкой, схемой признаков и происхождением данных; ответы после загрузки сверены.

## Сопоставимое сравнение

| Метод | Macro-F1, среднее ± std по людям | Ложные замечания | Обнаруженные ошибки |
|---|---:|---:|---:|
{table}

Главная строка — автоматическая процедура выбора семейства и параметров **только по внутренним разбиениям**. Число выборов каждого семейства: {families}. Показ отдельных семейств не означает, что финальная модель выбрана по лучшему внешнему результату. Настроек всего {len(config['candidates'])}. Seed этого запуска: {config['seed']}; аугментации и нейросеть здесь не проверяются.

![Результаты по новым людям](people_metrics.png)

| Тестовый человек | Macro-F1 | Ложные замечания | Обнаруженные ошибки |
|---|---:|---:|---:|
{person_table}

Самая высокая доля ложных замечаний у {most_false_feedback['test_person']}: {most_false_feedback['false_feedback_rate']:.1%}; самая низкая обнаруживаемость ошибок у {least_detection['test_person']}: {least_detection['error_recall']:.1%}. Среднее не означает одинаковой пригодности для всех людей.

![Матрица ошибок по наблюдениям камер](confusion_matrix.png)

По объединённым наблюдениям F1 класса «колени внутрь» — {chosen['pooled_repeat_weighted']['per_class_f1'][1]:.3f}, класса «недостаточная глубина» — {chosen['pooled_repeat_weighted']['per_class_f1'][2]:.3f}. Основная слабость — распознавание сведения коленей и перенос между стилями выполнения. Это основание для следующего запланированного опыта с фазовыми/относительными признаками; не доказательство, что новая архитектура обязательно поможет.

## Как проверяли

Пять внешних разбиений: один человек полностью исключён из обучения. На остальных четверых — четыре внутренних разбиения по людям для выбора настроек. После выбора обучение на четырёх, затем проверка на пятом. Scaler, балансировка и выбор параметров не используют внешнего человека. Все камеры одного движения остаются вместе.

Метрики одной камеры учитываются с весом 1/3, чтобы физическое повторение имело общий вес 1; это оценка одиночного наблюдения, не голосование камер. Таблица выше сначала усредняет классы внутри человека, затем людей. Общий Macro-F1 по объединённым предсказаниям равен {chosen['pooled_repeat_weighted']['macro_f1']:.3f} и является другим способом усреднения.

Для проверки арифметики: в объединённых наблюдениях корректных выполнений — {counts['correct_executions']}, ложных замечаний — {counts['false_feedback']}; наблюдений ошибочных выполнений — {counts['error_executions']}, замечаний по ним — {counts['detected_errors']}, из них правильный тип ошибки — {counts['correct_error_type']}. Эти числа относятся к **камерам**, а не к отдельным независимым движениям. Взвешенные количества в JSON — эквиваленты физических повторений, иногда дробные.

Ориентиры проверялись при выборе внутренних вариантов; если подходящих не было, сохранялся исследовательский вариант с максимальным Macro-F1, без объявления ограничений достигнутыми. Никакой отказ в этом опыте не применялся: покрытие 100% подготовленных наблюдений. «Уверенность» не откалибрована; SVM использует оценку отступа, а не вероятность правильности.

## Ограничения и следующий шаг

- Три категории REHAB26 и пять категорий EC3D — разные задачи. Нельзя приписать улучшение модели, просто сравнив их итоговые числа.
- Только пять людей; дополнительные камеры и повторы не делают статистику независимой. Std по людям описательный; статистическая значимость улучшения не установлена.
- Все ракурсы присутствовали в обучающих людях. Это проверка нового человека при разных ракурсах, не тест неизвестного углового диапазона.
- 166 приседаний с другими ошибками остались вне основной трёхклассовой задачи. Отсутствие двух выбранных признаков не означает правильность всей техники.
- Используются уже вырезанные авторами MediaPipe world-последовательности. Raw RGB, автоматическая сегментация и другой экстрактор позы ещё не проверены; `last_frame` явно включается по принятому соглашению, точная семантика границы не подтверждена RGB-просмотром.
- Следующий опыт по ранее утверждённому плану — сравнение глобальной геометрии с относительными/фазовыми и временными признаками. Версию и правила сравнения фиксировать до запуска. Эти внешние результаты уже просмотрены: дальнейшая ручная разработка на них является исследовательской, для сильной статьи требуется независимое подтверждение.

## Запуск и файлы

```bash
python scripts/download_rehab26.py
python scripts/rehab26_baseline.py --prepare-only
python scripts/visualize_rehab26.py
python scripts/rehab26_baseline.py
python -m unittest discover -s tests -v
```

Суммарный измеренный запуск подготовки/подбора/оценки/финального обучения: {runtime['seconds']:.1f} с на текущем CPU, без скачивания и построения галерей; GPU не использовался. Это время зависит от железа.

Пути к моделям и исходному коду записаны в `run.json` журнала запуска. Параметры финальной модели выбирались отдельной проверкой по людям на всех данных. Она обучена на всех пяти людях, собственного независимого теста этих весов нет, порог отказа не настроен. Ещё сохранены {runtime['saved_fold_bundles']} моделей внешних разбиений. Каталог моделей исключён из Git, модели восстанавливаются запуском скрипта.

Применение без обучения: `python scripts/predict_rehab26.py repetition.npy --model <путь_из_журнала>/squat_geometry.joblib`. Вход — **одно полное повторение** формата `(кадры,33,4)` MediaPipe world xyz + visibility; EC3D и RGB напрямую не подходят. Файлы joblib загружаются только из доверенного источника.

Артефакты: [метрики](metrics.json), [предсказания камер](fold_predictions.csv), [разбиения](splits.json), [внутренний подбор](inner_search.json), [финальный выбор](final_training_selection.json), [зафиксированный запуск](run_config.json), [аудит поз](data_audit.json), [схема признаков](feature_metadata.json), [галереи](pose_galleries/).
"""
    (root / "README.md").write_text(text, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, (key, title, bound) in zip(axes, (
        ("macro_f1", "Macro-F1: выше лучше", None),
        ("false_feedback_rate", "Ложные замечания: ниже лучше", .2),
        ("error_recall", "Обнаруженные ошибки: выше лучше", .8),
    )):
        ax.bar([r["test_person"] for r in people], [r[key] for r in people], color="#2c7da0")
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=.2)
        if bound is not None:
            ax.axhline(bound, color="#c1121f", linestyle="--", linewidth=1.5, label="Рабочий ориентир")
            ax.legend(fontsize=8)
    fig.savefig(root / "people_metrics.png", dpi=160)
    plt.close(fig)

    matrix = np.array(chosen["pooled_repeat_weighted"]["confusion_matrix_observations"])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    names = ["Корректно*", "Колени внутрь", "Малая глубина"]
    ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
           xlabel="Предсказанная категория", ylabel="Метка набора", title="Новый человек: все внешние предсказания")
    ax.tick_params(axis="x", rotation=15)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value}\n{normalized[i,j]:.0%}", ha="center", va="center", color="white" if normalized[i,j] > .5 else "black")
    fig.savefig(root / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(f"Отчёт и графики: {root}")


if __name__ == "__main__":
    main()
