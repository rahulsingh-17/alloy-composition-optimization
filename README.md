# Alloy Composition Optimization

This project focuses on predicting the optimal alloy composition for now **copper-based alloys** based only on user-defined mechanical and electrical property requirements.

## Overview

The system uses machine learning models to recommend alloy compositions that best match desired target properties, including:

- Electrical Conductivity  
- Hardness  
- Tensile Strength  
- Yield Strength  

## Models Used

Two regression models are implemented and compared:

- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  

These models are trained on a dataset of copper alloy compositions and their corresponding material properties.

## Input Methods 

### 2. Numeric Input

- Users enter values for each property required 

Example:
- Conductivity: 80 %IACS  
- Tensile Strength: 500 MPa  

## Workflow

1. User provides desired property values 
2. Input is processed and converted into structured format  
3. Trained ML models predict the optimal alloy composition  
4. Output includes recommended composition and expected properties  

## Dataset

- Based on copper alloy compositions  
- Includes:
  - Elemental composition (percentage of elements)  
  - Measured physical and mechanical properties  

## Goals

- Assist in materials design and optimization  
- Reduce trial-and-error in alloy development  
- Provide a data-driven approach for engineering decisions  

## Future Improvements

- Add more alloy systems (e.g., aluminum, steel)  
- Add keyword-based NLP with advanced language models  
- Integrate optimization algorithms (e.g., genetic algorithms)  
- Improve UI/UX of the web application  

## Requirements

- Python 3.x  
- scikit-learn  
- pandas  
- numpy

## How to Run

Follow these steps to run the project on your system:

1. Download or clone the project from GitHub  

2. Open the project folder in terminal   

3. Install required libraries
   
   pip install -r requirements.txt

4. Run the Flask app

   python app.py

5. Open your browser and go to:

   http://127.0.0.1:5000/

7. Enter the required values and get the predicted optimized alloy composition
