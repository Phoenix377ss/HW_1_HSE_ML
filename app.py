import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
from utils import CarPricePredictor

MODEL_PATH = Path(__file__).parent / "model.pkl"

RAW_FEATURES = [
    "name", "year", "km_driven", "fuel", "seller_type",
    "transmission", "owner", "mileage", "engine",
    "max_power", "torque", "seats",
]

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

if not MODEL_PATH.exists():
    st.error("Нет файла model.pkl")
    st.stop()

model = load_model()

st.title("Какая будет цена авто")
st.caption("Модель регрессии с препроцессингом фичей")

with st.form("car"):
    left, right = st.columns(2)

    with left:
        name = st.text_input("Название модели (бренд и модель)", value="Maruti Swift Dzire VDI")
        year = st.number_input("Год выпуска", min_value=1980, max_value=2025, value=2015)
        km_driven = st.number_input("Пробег, км", min_value=0, max_value=3_000_000, value=60_000, step=1000)
        mileage = st.number_input("Расход, kmpl", min_value=0.0, max_value=50.0, value=18.9, step=0.1)
        engine = st.number_input("Объём двигателя, CC", min_value=500, max_value=6000, value=1248, step=50)
        max_power = st.number_input("Мощность, bhp", min_value=20.0, max_value=1000.0, value=74.0, step=1.0)

    with right:
        fuel = st.selectbox("Топливо", ["Diesel", "Petrol", "CNG", "LPG"])
        seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
        owner = st.selectbox(
            "Владелец",
            ["First Owner", "Second Owner", "Third Owner",
             "Fourth & Above Owner", "Test Drive Car"],
        )
        seats = st.selectbox("Мест", [2, 4, 5, 6, 7, 8, 9, 14], index=2)
        
        torque_nm = st.number_input("Крутящий момент, Nm", min_value=0.0, max_value=2000.0, value=190.0, step=5.0)
        torque_rpm = st.number_input("Обороты макс. момента, rpm", min_value=0, max_value=10000, value=2000, step=100)

    submitted = st.form_submit_button("Предсказать цену")

if submitted:
    torque_str = f"{torque_nm:.0f}Nm@ {torque_rpm}rpm"
    
    car = pd.DataFrame([{
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,    
        "engine": engine,      
        "max_power": max_power, 
        "torque": torque_str,   
        "seats": seats,
    }])

    car = car[RAW_FEATURES]

    price = float(model.predict(car)[0])
    st.metric("Предсказанная цена", f"{price:,.0f} ₽")

st.subheader("Прогноз по файлу")
uploaded = st.file_uploader("CSV с автомобилями", type="csv")
if uploaded is None:
    st.info("Загрузите CSV с теми же колонками, что ждёт модель.")
else:
    batch = pd.read_csv(uploaded)
    missing = [c for c in RAW_FEATURES if c not in batch.columns]
    if missing:
        st.error(f"В файле нет колонок: {', '.join(missing)}")
    else:
        preds = model.predict(batch)
        batch = batch.copy()
        batch["predicted_price"] = preds
        st.metric("Средняя предсказанная цена", f"{batch['predicted_price'].mean():,.0f} ₽")
        st.dataframe(batch.sort_values("predicted_price", ascending=False), hide_index=True)
        st.download_button(
            "Скачать прогнозы",
            data=batch.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
            on_click="ignore",
        )


st.markdown("---")
st.subheader("Визуализация весов модели (топ-15 признаков)")


coefs = model.model.coef_
feat_names = model.feature_columns


weights_df = pd.DataFrame({
    "feature": feat_names,
    "coefficient": coefs
})
weights_df["abs_coef"] = weights_df["coefficient"].abs()
weights_df = weights_df.sort_values("abs_coef", ascending=False).head(15)


top_pos = weights_df[weights_df["coefficient"] > 0].sort_values("coefficient", ascending=True)
top_neg = weights_df[weights_df["coefficient"] < 0].sort_values("coefficient", ascending=False)


st.bar_chart(
    weights_df.set_index("feature")["coefficient"],
    horizontal=True,
    color="#4CAF50",
    width="stretch",
)


st.caption(
    "Положительный коэффициент увеличивает предсказанную цену, "
    "отрицательный — уменьшает. Значения даны в логарифмическом масштабе."
)

st.write("**Топ-5 самых влиятельных признаков:**")
for _, row in weights_df.head(5).iterrows():
    sign = "+" if row["coefficient"] > 0 else ""
    st.write(f"- `{row['feature']}`: {sign}{row['coefficient']:.4f}")