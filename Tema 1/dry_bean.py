from pandas import read_csv
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os

os.chdir("......")

filename = "Dry_Bean_Dataset.csv"
dataset = read_csv(filename)

# Convertim in array numpy
array = dataset.values

# Caracteristici: coloanele 0-15, Etichete: coloana 16
X = array[:, 0:16]
Y = array[:, 16]

# Impartire in set de antrenament 80% / test 20%
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=1
)

# Normalizare MinMax
minmaxscale = MinMaxScaler().fit(X_train)
X_train = minmaxscale.transform(X_train)
X_validation = minmaxscale.transform(X_validation)

# Determinarea automata a valorii optime a lui k folosind cross-validare 10-fold
k_values = list(range(1, 31))  # Testam k de la 1 la 30
cv_scores = []

print("k\tCV Accuracy")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_result = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    mean_score = cv_result.mean()
    cv_scores.append(mean_score)
    print(f"{k}\t{mean_score:.4f}")

best_k = k_values[cv_scores.index(max(cv_scores))]
print(f"\nCel mai bun k pe baza CV: {best_k} cu acuratete {max(cv_scores):.4f}")

# Plot acuratetea CV in functie de k
plt.figure(figsize=(10,6))
plt.plot(k_values, cv_scores, marker='o', linestyle='-', color='b')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k = {best_k}')
plt.title('kNN: CV Accuracy vs Numarul de vecini (k)')
plt.xlabel('Numarul de vecini (k)')
plt.ylabel('Acuratete Cross-Validata')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

# Antrenarea modelului final cu k optim
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluarea modelului final
print("Acuratete pe setul de validare:", accuracy_score(Y_validation, predictions))
print("Matrice de confuzie:\n", confusion_matrix(Y_validation, predictions))
print("Raport de clasificare:\n", classification_report(Y_validation, predictions))
