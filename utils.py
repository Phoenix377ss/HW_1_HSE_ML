import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class CarPricePredictor:
    def __init__(self, alpha=10.0, min_brand_freq=10):
        self.alpha = alpha
        self.min_brand_freq = min_brand_freq
        self.medians = {}
        self.top_brands = set()
        self.km_cap = None
        self.torque_cap = None
        self.ohe_seats = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.ohe_cat = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.feature_columns = []

    def _parse_raw(self, df):
        data = df.copy()
        for col in ['mileage', 'engine', 'max_power']:
            data[col] = data[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

        s = data['torque'].astype(str).str.replace(',', '')
        data['torque'] = s.str.extract(r'^(\d+\.?\d*)').astype(float)
        data.loc[s.str.contains('kgm', case=False, na=False), 'torque'] *= 9.80665
        data['max_torque_rpm'] = s.str.extract(r'(\d+\.?\d*)\s*rpm', flags=2).astype(float)
        return data

    def _build_features(self, df, is_train=False):
        raw_name_lower = df['name'].astype(str).str.lower()
        brand = raw_name_lower.str.split().str.get(0)

        data = self._parse_raw(df)

        num_cols_to_fill = ['mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']
        for col in num_cols_to_fill:
            if is_train:
                self.medians[col] = data[col].median()
            data[col] = data[col].fillna(self.medians[col])

        data['engine'] = data['engine'].astype(int)
        data['seats'] = data['seats'].astype(int)

        if is_train:
            brand_counts = brand.value_counts()
            self.top_brands = set(brand_counts[brand_counts >= self.min_brand_freq].index)

        data['name'] = brand.where(brand.isin(self.top_brands), 'other')

        if is_train:
            seats_arr = self.ohe_seats.fit_transform(data[['seats']])
        else:
            seats_arr = self.ohe_seats.transform(data[['seats']])
        seats_df = pd.DataFrame(seats_arr, columns=self.ohe_seats.get_feature_names_out(['seats']), index=data.index)
        data = pd.concat([data.drop(columns=['seats']), seats_df], axis=1)

        cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
        if is_train:
            cat_arr = self.ohe_cat.fit_transform(data[cat_cols])
        else:
            cat_arr = self.ohe_cat.transform(data[cat_cols])
        cat_df = pd.DataFrame(cat_arr, columns=self.ohe_cat.get_feature_names_out(cat_cols), index=data.index)
        data = pd.concat([data.drop(columns=cat_cols), cat_df], axis=1)

        data['power_per_engine'] = data['max_power'] / (data['engine'] + 1e-5)
        data['age'] = 2020 - data['year']
        data['age_sq'] = data['age'] ** 2
        data['km_per_year'] = data['km_driven'] / (data['age'] + 1)
        data['power_x_year'] = data['max_power'] * data['age']

        lux_brands = {'bmw', 'mercedes-benz', 'audi', 'volvo', 'jaguar', 'land', 'lexus', 'porsche', 'jeep'}
        data['is_luxury'] = brand.isin(lux_brands).astype(int)
        data['is_top_trim'] = raw_name_lower.str.contains(r'\b(?:zxi|zdi|zx|vx|sx|alpha|titanium|top|awd|4wd|4x4|quattro)\b', regex=True).astype(int)
        data['owner_3plus'] = df['owner'].isin(['Third Owner', 'Fourth & Above Owner']).astype(int)
        data['first_owner_dealer'] = ((df['owner'] == 'First Owner') & (df['seller_type'].isin(['Dealer', 'Trustmark Dealer']))).astype(int)

        if is_train:
            self.km_cap = data['km_driven'].quantile(0.99)
            self.torque_cap = data['torque'].quantile(0.99)

        data['km_driven'] = np.clip(data['km_driven'], None, self.km_cap)
        data['torque'] = np.clip(data['torque'], None, self.torque_cap)

        for col in ['km_driven', 'max_power', 'engine', 'torque']:
            data[f'log_{col}'] = np.log1p(np.maximum(data[col], 0))

        return data

    def fit(self, df_train, target_col='selling_price'):
        X = df_train.drop(columns=[target_col])
        y = np.log1p(df_train[target_col])
        X_feat = self._build_features(X, is_train=True)
        self.feature_columns = X_feat.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_feat)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, df_new):
        X = df_new.drop(columns=['selling_price'], errors='ignore')
        X_feat = self._build_features(X, is_train=False)
        X_feat = X_feat.reindex(columns=self.feature_columns, fill_value=0)
        X_scaled = self.scaler.transform(X_feat)
        pred_log = self.model.predict(X_scaled)
        return np.expm1(pred_log)
