from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib



# قراءة البيانات
df = pd.read_csv(r"C:\Users\abnso\OneDrive\Desktop\lap_test\data\cleaned_data.csv")

# تحديد الخصائص والهدف
X = df.drop(columns=['Price_euros'])
y = np.log(df['Price_euros'])  # أخذ اللوغ لتحسين التدريب

# تجهيز الـ ColumnTransformer للأعمدة الكاتيجوريك
trans = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first',handle_unknown="ignore"), [0,1,7,10,11])
], remainder='passthrough')

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# نموذج RandomForest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=3,
    max_samples=0.5,
    max_features=0.75,
    max_depth=15
)

# إنشاء Pipeline
pipe = Pipeline([
    ('step1', trans),
    ('step2', model)
])


# تدريب النموذج
pipe.fit(X_train, y_train)

# التوقعات
y_pred = pipe.predict(X_test)

# عكس اللوغ للحصول على القيم الأصلية
y_pred_original = np.exp(y_pred)
y_test_original = np.exp(y_test)

# مقارنة أول 5 قيم فعلية ومتوقعة
comparison = pd.DataFrame({
    'Actual': y_test_original[:5].values,
    'Predicted': y_pred_original[:5]
})
print("Comparison of first 5 predictions:")
print(comparison)

# تقييم النموذج
print('\nR2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

joblib.dump(pipe, r"G:\programing\Model_app\ML\model\pipe.pkl")   
joblib.dump(X.columns.tolist(), r"G:\programing\Model_app\ML\model\columns.pkl")

