import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


df = pd.read_csv("data/synthetic_loan_approval.csv")
feature_cols = [
    "Age", "Gender", "MaritalStatus", "EducationLevel", "EmploymentStatus",
    "AnnualIncome", "LoanAmountRequested", "PurposeOfLoan",
    "CreditScore", "ExistingLoansCount", "LatePaymentsLastYear"
]
X = df[feature_cols]
y = df["LoanApproved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = ["Age", "AnnualIncome", "LoanAmountRequested", "CreditScore", "ExistingLoansCount", "LatePaymentsLastYear"]
categorical_features = ["Gender", "MaritalStatus", "EducationLevel", "EmploymentStatus", "PurposeOfLoan"]

#xgb
xgb_preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)

xgb_clf = Pipeline(steps=[
    ("preprocessor", xgb_preprocessor),
    ("model", xgb_model)
])

xgb_clf.fit(X_train, y_train)
roc_xgb = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])
print(f"XGBoost ROC-AUC: {roc_xgb:.4f}")

low_score_val = int(df["CreditScore"].quantile(0.25))

joblib.dump({
    "pipeline": xgb_clf,
    "feature_cols": feature_cols,
    "neutral_score": low_score_val
}, "models/xgb_model.pkl")

# NN
nn_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

X_train_nn = nn_preprocessor.fit_transform(X_train)
X_test_nn = nn_preprocessor.transform(X_test)
input_dim = X_train_nn.shape[1]

class LoanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LoanNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

train_loader = DataLoader(LoanDataset(X_train_nn, y_train), batch_size=64, shuffle=True)
model = LoanNN(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(20):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_nn, dtype=torch.float32)
    preds = model(X_test_tensor).numpy()
    roc_nn = roc_auc_score(y_test, preds)
print(f"NN ROC-AUC: {roc_nn:.4f}")

joblib.dump(nn_preprocessor, "models/nn_preprocessor.pkl")
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": input_dim
}, "models/nn_model.pt")

print("Модели успешно сохранены.")
