import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# load the dataset
df = pd.read_csv("Cu_alloys_database_2023_06_06.csv", sep=";", encoding="latin1")

# these are the elements we want to predict (output/composition)
elements = ["Cu", "Ni", "Si", "Cr", "Mg", "Al", "Zr", "Ti", "Co", "Fe", "Zn", "Sn"]

# these are the properties user will give as input
properties = [
    "Hardness (HV)",
    "Electrical conductivity (%IACS)",
    "Ultimate tensile strength (MPa)",
    "Yield strength (MPa)"
]

# drop rows where all 4 properties are missing
df = df.dropna(subset=properties, how="all")

# fill missing property values with column mean
for col in properties:
    df[col] = df[col].fillna(df[col].mean())

# fill missing element values with 0
for col in elements:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# input features = the 4 properties
X = df[properties].values

# we will train one model per element
print("Training models...")
print("="*50)

models = {}

for element in elements:
    if element not in df.columns:
        continue

    y = df[element].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  {element:5s}  ->  R2: {r2:.4f}   RMSE: {rmse:.4f}")
    models[element] = model

# save all models to a file
joblib.dump(models, "composition_models.pkl")
print("\nModels saved to composition_models.pkl")
print("="*50)


# prediction function
# user gives property values, model predicts the composition
def predict_composition(hardness, conductivity, tensile_strength, yield_strength):

    # load models
    saved_models = joblib.load("composition_models.pkl")

    # prepare input
    user_input = np.array([[hardness, conductivity, tensile_strength, yield_strength]])

    print("\n" + "="*50)
    print("INPUT PROPERTIES:")
    print(f"  Hardness               : {hardness} HV")
    print(f"  Electrical Conductivity: {conductivity} %IACS")
    print(f"  Tensile Strength       : {tensile_strength} MPa")
    print(f"  Yield Strength         : {yield_strength} MPa")
    print("="*50)
    print("\nPREDICTED ALLOY COMPOSITION (wt%):")
    print("-"*35)

    composition = {}
    for element, model in saved_models.items():
        predicted_wt = model.predict(user_input)[0]
        predicted_wt = max(0, round(predicted_wt, 4))   # no negative values
        composition[element] = predicted_wt

    # sort by highest wt% first
    composition = dict(sorted(composition.items(), key=lambda x: x[1], reverse=True))

    for elem, wt in composition.items():
        if wt > 0.001:   # only show elements that are actually present
            print(f"  {elem:5s}: {wt:.4f} wt%")

    total = sum(composition.values())
    print("-"*35)
    print(f"  Total : {total:.4f} wt%")
    print("="*50)

    return composition


# ── RUN PREDICTION ────────────────────────────────────────────────────────────
# change these values to whatever properties you want
hardness         = 250     # HV
conductivity     = 45      # %IACS
tensile_strength = 700     # MPa
yield_strength   = 500     # MPa

predict_composition(hardness, conductivity, tensile_strength, yield_strength)
