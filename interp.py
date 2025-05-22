import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Load all 50 entries from both files
info = pd.read_csv('info.csv').head(100)
results = pd.read_csv('results.csv').head(100)

# Merge on 'ID'
df = pd.merge(info, results, on='ID')

# Group by Age_group and calculate mean systolic and diastolic pressure
age_bp = df.groupby('Age_group')[['systolic_pressure', 'diastolic_pressure']].mean().reset_index()

# Ensure Age_group is sorted and numeric for interpolation
age_bp = age_bp.sort_values('Age_group')
x = age_bp['Age_group']
if not np.issubdtype(x.dtype, np.number):
    x = np.arange(len(x))  # fallback if Age_group is categorical

xnew = np.linspace(x.min(), x.max(), 300)

# Interpolate systolic and diastolic
spl_sys = make_interp_spline(x, age_bp['systolic_pressure'], k=3)
sys_smooth = spl_sys(xnew)
spl_dia = make_interp_spline(x, age_bp['diastolic_pressure'], k=3)
dia_smooth = spl_dia(xnew)

plt.figure(figsize=(10, 6))
# Remove these lines to NOT connect the points:
# plt.plot(xnew, sys_smooth, label='Systolic Pressure (smoothed)')
# plt.plot(xnew, dia_smooth, label='Diastolic Pressure (smoothed)')

plt.scatter(x, age_bp['systolic_pressure'], color='blue', marker='o', label='Systolic Pressure')
plt.scatter(x, age_bp['diastolic_pressure'], color='orange', marker='s', label='Diastolic Pressure')

# Add trendlines (linear regression)
p_sys = Polynomial.fit(x, age_bp['systolic_pressure'], 1)
p_dia = Polynomial.fit(x, age_bp['diastolic_pressure'], 1)
plt.plot(xnew, p_sys(xnew), 'b--', label='Systolic Trend')
plt.plot(xnew, p_dia(xnew), color='orange', linestyle='dashed', label='Diastolic Trend')

plt.xlabel('Age Group')
plt.ylabel('Blood Pressure (mmHg)')
plt.title('Average Blood Pressure by Age Group (Trend Only)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()