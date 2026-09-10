# Conversion Prediction Model

Модель предсказывает вероятность целевого действия (конверсии) пользователя на сайте по данным сессии Google Analytics (utm-метки, устройство, гео, время визита).

Целевое действие — совершение одного из следующих событий в рамках сессии:
```text
sub_car_claim_click
sub_car_claim_submit_click
sub_open_dialog_click
sub_custom_question_submit_click
sub_call_number_click
sub_callback_submit_click
sub_submit_success
sub_car_request_submit_click
```
## Структура проекта
```text
pipeline.py       # загрузка данных, feature engineering, сборка sklearn-пайплайна
train.py          # обучение модели, сохранение ga_model.pkl
main.py           # FastAPI-сервис для инференса
ga_model.pkl      # обученная модель (создаётся train.py)
ga_sessions.*   # исходные данные о сессиях (не входит в репозиторий)
ga_hits*.*   # исходные данные о событиях (не входит в репозиторий)
requirements.txt
```
## Установка
```text
bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

Минимальный набор зависимостей:

fastapi
uvicorn
pandas
numpy
scikit-learn
dill
pyarrow
```
## Данные
```text
Перед обучением положите в корень проекта файлы с сессиями и событиями. 
Загрузчик данных (read_data_file в pipeline.py) находит их автоматически по префиксу имени, независимо от:
формата файла — поддерживаются .parquet, .pkl, .csv (проверяются именно в этом порядке — если рядом лежит несколько форматов одного файла, будет использован первый найденный);
суффикса/номера в имени — подойдут ga_hits-001.parquet, ga_hits-002.parquet, ga_hits_final.csv и т.п.

Ожидаемые префиксы:

ga_sessions... — данные о сессиях (utm-метки, устройство, гео, дата/время визита и т.д.)
ga_hits... — данные о событиях (session_id, event_action), используются для формирования таргета

Если файл не найден ни в одном из поддерживаемых форматов, train.py завершится понятной ошибкой FileNotFoundError с указанием, какие шаблоны имён проверялись.

Если данные лежат в .csv, обратите внимание: этот формат не хранит типы данных так строго, как .parquet/.pkl — при необходимости стоит проверить df.dtypes после загрузки.
```

## Обучение модели
```text
bash
python train.py


Скрипт:

1. Загружает `ga_sessions.parquet` и `ga_hits-001.parquet`, формирует бинарный таргет (`target`) по наличию целевых событий в сессии.
2. Делит данные на train/test (70/30, стратификация по таргету).
3. Обучает пайплайн: feature engineering → очистка данных → группировка редких категорий (top-N) → отсечение выбросов по высоте экрана → отбор признаков → OneHot/импутация → `DecisionTreeClassifier`.
4. Выводит ROC-AUC на train и test.
5. Сохраняет модель вместе с метаданными в `ga_model.pkl` (сериализация через `dill`, чтобы сохранить кастомные трансформеры).

Пример вывода:

Загрузка GA Hits...
  Найден файл: ga_hits-001.parquet
GA Hits загружены.
Target сформирован.
Загрузка GA Sessions...
  Найден файл: ga_sessions.parquet
GA Sessions: (XXXXXX, XX)
Итоговый датасет: (XXXXXX, XX)
Доля target=1: 0.XXXX

Train: (XXXXXX, XX)
Test:  (XXXXXX, XX)

Обучение Decision Tree...
Время обучения: XX.X сек

Train ROC-AUC: 0.XXXX
Test ROC-AUC:  0.XXXX
ROC-AUC gap:   0.XXXX
Модель сохранена: /path/to/ga_model.pkl
```
## Признаки, используемые моделью
```text
Числовые: visit_number, has_keyword, is_russia, is_presence_city, visit_weekday, visit_hour, screen_width, screen_height.

Категориальные (после группировки редких значений в other):** utm_medium, device_category, device_os, device_browser, geo_city_grouped, utm_source_grouped, utm_campaign_grouped, device_brand_grouped, utm_adcontent_grouped.

Признаки session_id, client_id, visit_date, visit_time, device_screen_resolution и исходные (негруппированные) категориальные поля используются только на промежуточных этапах и не подаются в модель напрямую.
```
## Запуск API
```text
bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000


Документация Swagger будет доступна по адресу: `http://localhost:8000/docs`
```
### Эндпоинты

| Метод | Путь      | Описание                                            |
|-------|-----------|-----------------------------------------------------|
| GET   | /status   | Проверка работоспособности сервиса                  |
| GET   | /version  | Метаданные модели (название, версия, дата обучения) |
| POST  | /predict  | Предсказание вероятности конверсии                  |

### Пример запроса POST /predict
```text
json
{
  "session_id": "1234567890.1234567890",
  "client_id": "987654321.1234567890",
  "visit_date": "2021-10-15",
  "visit_time": "14:32:10",
  "visit_number": 1,
  "utm_source": "google",
  "utm_medium": "cpc",
  "utm_campaign": "spring_promo",
  "utm_adcontent": "banner_1",
  "utm_keyword": "car insurance",
  "device_category": "mobile",
  "device_os": "Android",
  "device_brand": "Samsung",
  "device_browser": "Chrome",
  "device_model": "SM-G960F",
  "geo_country": "Russia",
  "geo_city": "Moscow",
  "device_screen_resolution": "412x915"
}
```

### Пример ответа
```text
json
{
  "session_id": "1234567890.1234567890",
  "prediction": 1,
  "probability": 0.7421
}


prediction — бинарный класс (0/1), полученный по порогу 0.5 от probability. probability — вероятность целевого действия, оценённая моделью.
```
## Важно
```text
Все поля формы, кроме session_id, client_id, visit_date, visit_time, visit_number, являются опциональными (None по умолчанию) — пропуски обрабатываются пайплайном автоматически.
main.py и train.py используют одни и те же классы трансформеров из pipeline.py, поэтому обработка данных при обучении и при инференсе идентична.
Загрузка сырых данных (load_data() в pipeline.py) не завязана на конкретное имя файла или формат — можно свободно менять номер файла событий (-001, -002, ...) или формат (.parquet/.pkl/.csv), не трогая код.
Модель - DecisionTreeClassifier с class_weight="balanced", max_depth=10, min_samples_leaf=20 (ограничения глубины/листьев снижают переобучение при выраженном дисбалансе классов).
```
