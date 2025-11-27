import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Setăm un stil vizual mai plăcut pentru plot-uri
sns.set_theme(style="whitegrid")

# --- 1. Încărcarea Setului de Date ---
try:
    data = pd.read_csv('Housing.csv')
except FileNotFoundError:
    print("Eroare: Fișierul 'Housing.csv' nu a fost găsit.")
    exit()
print("Setul de date 'Housing.csv' a fost încărcat cu succes.")

# --- PLOT 1: Distribuția Variabilei Țintă (Price) ---
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True, bins=30)
plt.title('Figura 2: Distribuția Prețurilor (Variabila Țintă)', fontsize=16)
plt.xlabel('Preț')
plt.ylabel('Frecvență')
plt.tight_layout()
plt.savefig('plot_distributie_pret.png', dpi=300)
print("Graficul 'plot_distributie_pret.png' a fost salvat.")


# --- 2. Preprocesarea Datelor ---
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})
print("Coloanele binare 'yes'/'no' au fost transformate în 1/0.")

data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)
print("Coloana 'furnishingstatus' a fost transformată prin One-Hot Encoding.")

# --- PLOT 2: Heatmap-ul Corelațiilor ---
plt.figure(figsize=(14, 10))
sns.heatmap(data.corr(), annot=True, cmap='vlag', fmt='.2f', annot_kws={"size": 8})
plt.title('Figura 3: Heatmap Corelații (inclusiv trăsături preprocesate)', fontsize=16)
plt.tight_layout()
plt.savefig('plot_heatmap_corelatii.png', dpi=300)
print("Graficul 'plot_heatmap_corelatii.png' a fost salvat.")


# --- 2c. Separarea trăsăturilor (X) de țintă (y) ---
X = data.drop('price', axis=1)
y = data['price']
feature_names = X.columns
print(f"Numărul total de trăsături după preprocesare: {len(feature_names)}")

# --- 3. Împărțirea și Standardizarea Datelor ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Datele au fost împărțite și standardizate.")

# --- 4. Antrenarea Modelelor ---
results = {}

# Model 1: OLS
ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)
y_pred_train_ols = ols_model.predict(X_train_scaled)
y_pred_test_ols = ols_model.predict(X_test_scaled)
results['OLS'] = {
    'RMSE Train': np.sqrt(mean_squared_error(y_train, y_pred_train_ols)),
    'RMSE Test': np.sqrt(mean_squared_error(y_test, y_pred_test_ols)),
    'R2 Test': r2_score(y_test, y_pred_test_ols),
    'Alpha': 'N/A'
}

# Model 2: Ridge (L2)
alphas_ridge = 10**np.linspace(-2, 4, 100)
ridge_cv = RidgeCV(alphas=alphas_ridge, store_cv_results=True, scoring="neg_mean_squared_error")
ridge_cv.fit(X_train_scaled, y_train)
y_pred_train_ridge = ridge_cv.predict(X_train_scaled)
y_pred_test_ridge = ridge_cv.predict(X_test_scaled)
results['Ridge (L2)'] = {
    'RMSE Train': np.sqrt(mean_squared_error(y_train, y_pred_train_ridge)),
    'RMSE Test': np.sqrt(mean_squared_error(y_test, y_pred_test_ridge)),
    'R2 Test': r2_score(y_test, y_pred_test_ridge),
    'Alpha': round(ridge_cv.alpha_, 2)
}

# Model 3: Lasso (L1)
alphas_lasso = 10**np.linspace(-2, 4, 100)
lasso_cv = LassoCV(alphas=alphas_lasso, cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
y_pred_train_lasso = lasso_cv.predict(X_train_scaled)
y_pred_test_lasso = lasso_cv.predict(X_test_scaled)
results['Lasso (L1)'] = {
    'RMSE Train': np.sqrt(mean_squared_error(y_train, y_pred_train_lasso)),
    'RMSE Test': np.sqrt(mean_squared_error(y_test, y_pred_test_lasso)),
    'R2 Test': r2_score(y_test, y_pred_test_lasso),
    'Alpha': round(lasso_cv.alpha_, 2)
}
print("Toate modelele au fost antrenate.")

# --- 5. Afișarea Rezultatelor (Tabel) ---
print("\n--- Tabel 1: Comparația Performanței Modelelor ---")
results_df = pd.DataFrame(results).T
print(results_df.to_markdown(floatfmt=(".0f", ".0f", ".4f", "g")))

# --- 6. Analiza Coeficienților (Text) ---
print("\n--- Analiza Coeficienților ---")
num_coef_total = len(ols_model.coef_)
print(f"Număr total de trăsături: {num_coef_total}")
num_coef_ridge_zero = np.sum(np.abs(ridge_cv.coef_) < 1e-10)
print(f"Model Ridge: {num_coef_ridge_zero} din {num_coef_total} coeficienți sunt practic zero.")
num_coef_lasso_zero = np.sum(lasso_cv.coef_ == 0)
print(f"Model Lasso: {num_coef_lasso_zero} din {num_coef_total} coeficienți sunt exact zero.")


# --- 7. Generarea Graficelor ---

# PLOT 3: Comparația Coeficienților
coefs = pd.DataFrame({
    'Trăsătură': feature_names,
    'OLS': ols_model.coef_,
    'Ridge': ridge_cv.coef_,
    'Lasso': lasso_cv.coef_
})
coefs_melted = coefs.melt(id_vars='Trăsătură', var_name='Model', value_name='Magnitudine Coeficient')
plt.figure(figsize=(10, 8))
sns.barplot(data=coefs_melted, y='Trăsătură', x='Magnitudine Coeficient', hue='Model', orient='h')
plt.title('Figura 1: Comparația Magnitudinii Coeficienților', fontsize=16)
plt.xlabel('Magnitudine Coeficient (pe date standardizate)')
plt.ylabel('Trăsătură')
plt.legend(title='Model')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('grafic_comparatie_coeficienti_housing.png', dpi=300)
print("\nGraficul 'grafic_comparatie_coeficienti_housing.png' a fost salvat.")

# --- PLOT 4: Curba de Validare RIDGE ---
plt.figure(figsize=(10, 6))
# === LINIA CORECTATĂ (pentru a trata 'cv_results_' ca pe un array) ===
# 'cv_results_' este un array numpy de forma (n_samples, n_alphas)
# Facem media pe axa 0 (pe samples) pentru a obține eroarea medie pentru fiecare alpha
# Rezultatele sunt 'neg_mean_squared_error', deci le înmulțim cu -1 pentru a obține MSE
mse_mean_ridge = np.mean(-ridge_cv.cv_results_, axis=0)
# =====================================================================
plt.plot(ridge_cv.alphas, mse_mean_ridge, linewidth=2)
plt.xscale('log')
plt.xlabel('Alpha (Scală Logaritmică)', fontsize=12)
plt.ylabel('Eroare Medie Pătratică (din Cross-Validation)', fontsize=12)
plt.title('Figura 4: Curba de Validare Ridge (Selectarea Alpha)', fontsize=16)
optim_alpha = ridge_cv.alpha_
optim_mse = -ridge_cv.best_score_
plt.axvline(optim_alpha, color='red', linestyle='--', label=f'Alpha Optim = {optim_alpha:.2f}')
plt.axhline(optim_mse, color='gray', linestyle=':', label=f'MSE Minim = {optim_mse:.0f}')
plt.legend()
plt.tight_layout()
plt.savefig('plot_validare_ridge.png', dpi=300)
print("Graficul 'plot_validare_ridge.png' a fost salvat.")

# --- PLOT 5: Curba de Validare LASSO ---
plt.figure(figsize=(10, 6))
# lasso_cv.mse_path_ are forma (n_alphas, n_folds), facem media pe axa 1
mse_mean_lasso = np.mean(lasso_cv.mse_path_, axis=1)
plt.plot(lasso_cv.alphas_, mse_mean_lasso, linewidth=2, color='orange')
plt.xscale('log')
plt.xlabel('Alpha (Scală Logaritmică)', fontsize=12)
plt.ylabel('Eroare Medie Pătratică (din Cross-Validation)', fontsize=12)
plt.title('Figura 5: Curba de Validare Lasso (Selectarea Alpha)', fontsize=16)
plt.axvline(lasso_cv.alpha_, color='red', linestyle='--', label=f'Alpha Optim = {lasso_cv.alpha_:.2f}')
plt.axhline(np.min(mse_mean_lasso), color='gray', linestyle=':', label=f'MSE Minim = {np.min(mse_mean_lasso):.0f}')
plt.legend()
plt.tight_layout()
plt.savefig('plot_validare_lasso.png', dpi=300)
print("Graficul 'plot_validare_lasso.png' a fost salvat.")


# --- PLOT 6: Grafic Actual vs. Prezise (Reziduale) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_test_ols, alpha=0.6, label=f'OLS (R2: {r2_score(y_test, y_pred_test_ols):.3f})')
sns.scatterplot(x=y_test, y=y_pred_test_lasso, alpha=0.6, label=f'Lasso (R2: {r2_score(y_test, y_pred_test_lasso):.3f})')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Linie Ideală (Actual = Prezise)')
plt.xlabel('Preț Actual (Date de Testare)', fontsize=12)
plt.ylabel('Preț Prezise de Model', fontsize=12)
plt.title('Figura 6: Performanța Modelului (Preț Actual vs. Prezise)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('plot_actual_vs_predicted.png', dpi=300)
print("Graficul 'plot_actual_vs_predicted.png' a fost salvat.")

print("\n--- Procesare finalizată. 6 grafice au fost salvate. ---")