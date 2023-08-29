import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Loaded the dataset
data = pd.read_csv('creditcard.csv')

# Separated features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Normalized numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applied SMOTE to balance the classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialized the Random Forest classifier
model = RandomForestClassifier(random_state=42)

# Trained the model
model.fit(X_train, y_train)

# Made predictions
y_pred = model.predict(X_test)

# Evaluated the model
report = classification_report(y_test, y_pred)
print(report)
