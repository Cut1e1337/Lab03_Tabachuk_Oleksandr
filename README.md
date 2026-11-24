# Лабораторна робота №3. Основи регресійного моделювання: економність авто (Auto MPG)

**Виконав:** Табачук Олександр Ігорович, група КН-42. **Мова:** Python.
Продовження [лабораторної №2](https://github.com/Cut1e1337/Lab02_Tabachuk_Oleksandr): той самий набір даних і та сама ціль `mpg`.

## Результат

Обрана модель `ln mpg ~ ln маси + ln потужності + рік + регіон` має на тесті RMSE 3,07 mpg (MAE 2,31) проти 7,98 (6,69) у константного прогнозу.

| Що | Вихідний файл | Зібраний файл |
|---|---|---|
| Звіт (він же виконуваний ноутбук) | [report.qmd](report.qmd) | [report.html](report.html) |
| Презентація Revealjs | [slides.qmd](slides.qmd) | [slides.html](slides.html) |

Відкрити онлайн: [звіт](https://cut1e1337.github.io/Lab03_Tabachuk_Oleksandr/report.html) · [слайди](https://cut1e1337.github.io/Lab03_Tabachuk_Oleksandr/slides.html). HTML-файли самодостатні, їх можна також завантажити й відкрити локально.

## Структура

- `data/raw/auto-mpg.data` — незмінений файл UCI
- `src/lab3.py` — підготовка, поділ, моделі, перехресна валідація
- `outputs/figures`, `outputs/tables`, `outputs/summary.json` — результати, створені звітом
- `references.bib` — бібліографія; `requirements.txt` — зафіксовані залежності

## Дані

Auto MPG, UCI (<https://archive.ics.uci.edu/dataset/9/auto+mpg>), CC BY 4.0, 398 × 9. Потрібна версія вже в репозиторії, завантажувати нічого не треба.

## Відтворення

Версії: **Python 3.12**, **Quarto 1.11.9** (встановлюється з `requirements.txt` як `quarto-cli`; можна й [інсталятором](https://quarto.org/docs/download/)).

Windows (PowerShell), з кореня репозиторію:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
$env:QUARTO_PYTHON = "$PWD\.venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
.venv\Lib\site-packages\quarto_cli\bin\quarto.exe render report.qmd
.venv\Lib\site-packages\quarto_cli\bin\quarto.exe render slides.qmd
```

Linux / macOS:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export QUARTO_PYTHON="$PWD/.venv/bin/python"
quarto render report.qmd && quarto render slides.qmd
```

Спершу звіт, потім слайди: вони читають збережені звітом таблиці й графіки. Результат — `report.html` і `slides.html`. Рендеринг звіту триває близько 1–2 хв.

> Якщо встановлено Quarto інсталятором, замість шляху до `quarto.exe` досить `quarto`.
