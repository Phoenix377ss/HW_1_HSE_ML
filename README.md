# Прогноз стоимости автомобиля (HW1 pro)

Учебный проект по машинному обучению: регрессия для предсказания цены подержанного автомобиля по его характеристикам. Включает EDA, feature engineering, обучение Ridge-регрессии, бизнес-метрику и интерактивное Streamlit-приложение.

## Что внутри

- **EDA**: анализ пропусков, дубликатов, распределений, выбросов; дашборд `ydata-profiling`.
- **Feature engineering**: парсинг текстовых признаков (`mileage`, `engine`, `max_power`, `torque`), бренд-фичи, `age`, `km_per_year`, `power_per_engine`, флаг люксовости, топ-комплектации, one-hot для категорий и `seats`.
- **Модель**: `Ridge` (alpha=10) на `log1p(selling_price)`, с `StandardScaler` и `OneHotEncoder`; всё завёрнуто в класс `CarPricePredictor` (`utils.py`).
- **Метрики**: R², MAE, RMSE; бизнес-метрика — доля предсказаний с ошибкой в пределах X% и средняя абсолютная ошибка в рублях.
- **Streamlit-приложение** (`app.py`): https://hw1hseml-5u5kwhpztaqj8adoqnneux.streamlit.app/
форма для одной машины, загрузка CSV для батча, скачивание прогнозов, визуализация топ-15 весов модели.

## Файлы

| Файл | Назначение |
|---|---|
| `app.py` | Streamlit-приложение (форма + батч + веса модели) |
| `utils.py` | класс `CarPricePredictor`: парсинг, фичи, обучение, предсказание |
| `model.pkl` | сохранённая обученная модель (pickle) |
| `requirements.txt` | зависимости для Streamlit Cloud |
| `sample.csv` | пример CSV для проверки батч-прогноза |

## Запуск локально

```bash
pip install -r requirements.txt
streamlit run app.py
