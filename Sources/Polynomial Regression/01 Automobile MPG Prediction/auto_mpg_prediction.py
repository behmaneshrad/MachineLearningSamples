import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# تنظیمات matplotlib برای نمایش فونت‌های فارسی
plt.rcParams['font.family'] = ['DejaVu Sans', 'Tahoma', 'Arial Unicode MS']
plt.style.use('seaborn-v0_8')

class AutoMPGPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, file_path=None):
        print("=" * 50)
        print("مرحله 1: بارگذاری داده‌ها")
        print("=" * 50)

        column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 
                        'weight', 'acceleration', 'model_year', 'origin', 'car_name']
        
        if file_path:
            self.data = pd.read_csv(file_path, names=column_names, sep=r'\s+', na_values='?')
        else:
            raise ValueError("file_path باید مشخص شود.")
        
        print(f"تعداد نمونه‌ها: {len(self.data)}")
        print(f"تعداد ویژگی‌ها: {len(self.data.columns)}")
        print("\nنمای کلی از داده‌ها:")
        print(self.data.head())
        print(f"\nاطلاعات کلی:")
        print(self.data.info())

    def explore_data(self):
        print("\n" + "=" * 50)
        print("مرحله 2: کاوش و تحلیل داده‌ها")
        print("=" * 50)
        
        print("آمار توصیفی:")
        print(self.data.describe())
        
        print(f"\nداده‌های مفقود:")
        missing_data = self.data.isnull().sum()
        print(missing_data[missing_data > 0])
        
        print(f"\nتوزیع کشور سازنده (origin):")
        print(self.data['origin'].value_counts())
        print("1: آمریکا، 2: اروپا، 3: ژاپن")
        
        self.plot_exploratory_analysis()

    def plot_exploratory_analysis(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('تحلیل اکتشافی داده‌های Auto MPG', fontsize=16, fontweight='bold')
        
        axes[0, 0].hist(self.data['mpg'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('توزیع مصرف سوخت (MPG)')
        
        axes[0, 1].scatter(self.data['weight'], self.data['mpg'], alpha=0.6, color='red')
        axes[0, 1].set_title('رابطه وزن و مصرف سوخت')
        
        valid_hp = self.data.dropna(subset=['horsepower'])
        axes[0, 2].scatter(valid_hp['horsepower'], valid_hp['mpg'], alpha=0.6, color='green')
        axes[0, 2].set_title('رابطه توان موتور و مصرف سوخت')
        
        cylinder_mpg = self.data.groupby('cylinders')['mpg'].mean()
        axes[1, 0].bar(cylinder_mpg.index, cylinder_mpg.values, color='orange', alpha=0.7)
        axes[1, 0].set_title('میانگین MPG بر اساس تعداد سیلندر')

        origin_labels = {1: 'آمریکا', 2: 'اروپا', 3: 'ژاپن'}
        origin_mpg = self.data.groupby('origin')['mpg'].mean()
        origin_names = [origin_labels[i] for i in origin_mpg.index]
        axes[1, 1].bar(origin_names, origin_mpg.values, color=['blue', 'red', 'green'], alpha=0.7)
        axes[1, 1].set_title('میانگین MPG بر اساس کشور سازنده')
        
        year_mpg = self.data.groupby('model_year')['mpg'].mean()
        actual_years = [1900 + y if y > 50 else 2000 + y for y in year_mpg.index]
        axes[1, 2].plot(actual_years, year_mpg.values, marker='o', color='purple')
        axes[1, 2].set_title('تغییرات میانگین MPG در طول زمان')
        
        plt.tight_layout()
        plt.show()

    def preprocess_data(self):
        print("\n" + "=" * 50)
        print("مرحله 3: پیش‌پردازش داده‌ها")
        print("=" * 50)

        processed_data = self.data.copy()
        processed_data = processed_data.drop('car_name', axis=1)

        if processed_data['horsepower'].isnull().sum() > 0:
            mean_hp = processed_data['horsepower'].mean()
            processed_data['horsepower'].fillna(mean_hp, inplace=True)
            print(f"مقادیر مفقود horsepower با میانگین {mean_hp:.2f} جایگزین شدند")

        processed_data['model_year'] = processed_data['model_year'].apply(
            lambda x: 1900 + x if x > 50 else 2000 + x
        )
        
        processed_data['power_to_weight'] = processed_data['horsepower'] / processed_data['weight']
        processed_data['displacement_per_cylinder'] = processed_data['displacement'] / processed_data['cylinders']

        self.processed_data = processed_data
        return processed_data

    def prepare_features(self):
        print("\n" + "=" * 50)
        print("مرحله 4: آماده‌سازی ویژگی‌ها")
        print("=" * 50)

        X = self.processed_data.drop('mpg', axis=1)
        y = self.processed_data['mpg']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_models(self):
        print("\n" + "=" * 50)
        print("مرحله 5: آموزش مدل‌ها")
        print("=" * 50)

        self.lr_model.fit(self.X_train_scaled, self.y_train)
        self.rf_model.fit(self.X_train, self.y_train)

    def evaluate_models(self):
        print("\n" + "=" * 50)
        print("مرحله 6: ارزیابی مدل‌ها")
        print("=" * 50)

        lr_pred_test = self.lr_model.predict(self.X_test_scaled)
        rf_pred_test = self.rf_model.predict(self.X_test)

        def metrics(y_true, y_pred):
            return {
                'MSE': mean_squared_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'R2': r2_score(y_true, y_pred)
            }

        lr_metrics = metrics(self.y_test, lr_pred_test)
        rf_metrics = metrics(self.y_test, rf_pred_test)

        print(f"رگرسیون خطی - R²: {lr_metrics['R2']:.4f}")
        print(f"Random Forest - R²: {rf_metrics['R2']:.4f}")

        self.plot_predictions(lr_pred_test, rf_pred_test)
        return {'lr_test': lr_metrics, 'rf_test': rf_metrics}

    def plot_predictions(self, lr_pred, rf_pred):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('پیش‌بینی vs مقدار واقعی', fontsize=14)

        axes[0].scatter(self.y_test, lr_pred, alpha=0.7, color='blue')
        axes[0].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        axes[0].set_title('رگرسیون خطی')

        axes[1].scatter(self.y_test, rf_pred, alpha=0.7, color='green')
        axes[1].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        axes[1].set_title('Random Forest')

        plt.tight_layout()
        plt.show()

    def predict_new_car(self, car_features):
        print("\n" + "=" * 50)
        print("مرحله 7: پیش‌بینی برای خودروی جدید")
        print("=" * 50)

        feature_names = ['cylinders', 'displacement', 'horsepower', 'weight', 
                         'acceleration', 'model_year', 'origin', 
                         'power_to_weight', 'displacement_per_cylinder']
        
        car_df = pd.DataFrame([car_features], columns=feature_names)
        car_scaled = self.scaler.transform(car_df)
        
        lr_pred = self.lr_model.predict(car_scaled)[0]
        rf_pred = self.rf_model.predict(car_df)[0]
        
        print(f"رگرسیون خطی: {lr_pred:.2f} MPG")
        print(f"Random Forest: {rf_pred:.2f} MPG")
        print(f"میانگین: {(lr_pred + rf_pred)/2:.2f} MPG")
        return lr_pred, rf_pred

def main():
    print("🚗 پروژه پیش‌بینی مصرف سوخت اتومبیل 🚗")
    print("=" * 60)

    predictor = AutoMPGPredictor()
    
    # 🔥 این خط تغییر کرده است:
    predictor.load_data(file_path='dataset/auto-mpg.data')
    
    predictor.explore_data()
    predictor.preprocess_data()
    predictor.prepare_features()
    predictor.train_models()
    metrics = predictor.evaluate_models()

    # مثال از خودروی جدید برای پیش‌بینی
    sample_car = [4, 2000, 120, 2500, 15.0, 2020, 3, 120/2500, 2000/4]
    predictor.predict_new_car(sample_car)

    print("\n" + "=" * 50)
    print("خلاصه نتایج")
    print("=" * 50)

    if metrics['lr_test']['R2'] > metrics['rf_test']['R2']:
        print(f"✅ بهترین مدل: رگرسیون خطی (R² = {metrics['lr_test']['R2']:.4f})")
    else:
        print(f"✅ بهترین مدل: Random Forest (R² = {metrics['rf_test']['R2']:.4f})")

if __name__ == "__main__":
    main()
