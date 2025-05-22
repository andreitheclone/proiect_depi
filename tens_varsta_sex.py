import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Load all 150 entries from both files
info = pd.read_csv('info.csv').head(150)
results = pd.read_csv('results.csv').head(150)

# Merge on 'ID'
df = pd.merge(info, results, on='ID')

# Split by Sex
df_male = df[df['Sex'] == 0]
df_female = df[df['Sex'] == 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, data, Sex in zip(axes, [df_male, df_female], ['Male', 'Female']):
    age_bp = data.groupby('Age_group')[['systolic_pressure', 'diastolic_pressure']].mean().reset_index()
    age_bp = age_bp.sort_values('Age_group')
    x = age_bp['Age_group']
    if not np.issubdtype(x.dtype, np.number):
        x = np.arange(len(x))
    xnew = np.linspace(x.min(), x.max(), 300)

    # Scatter plot
    ax.scatter(x, age_bp['systolic_pressure'], color='blue', marker='o', label='Systolic Pressure')
    ax.scatter(x, age_bp['diastolic_pressure'], color='orange', marker='s', label='Diastolic Pressure')

    # Trendlines (linear regression)
    if len(x) > 1:
        p_sys = Polynomial.fit(x, age_bp['systolic_pressure'], 1)
        p_dia = Polynomial.fit(x, age_bp['diastolic_pressure'], 1)
        ax.plot(xnew, p_sys(xnew), 'b--', label='Systolic Trend')
        ax.plot(xnew, p_dia(xnew), color='orange', linestyle='dashed', label='Diastolic Trend')

    ax.set_xlabel('Age Group')
    ax.set_title(f'Average Blood Pressure by Age Group ({Sex})')
    ax.grid(True)
    ax.legend()

axes[0].set_ylabel('Blood Pressure (mmHg)')
plt.tight_layout()
plt.show()