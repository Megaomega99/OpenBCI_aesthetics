import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0, 100, 1)
y = x**2
y = y + np.random.normal(0, 10, len(y))
y = y.astype(int)

# Create a DataFrame with the data
df = pd.DataFrame({'x': x, 'y': y})
df['y'] = df['y'].astype(int)
df['x'] = df['x'].astype(int)

df.to_csv('data.csv', index=True)
# Plot the data
plt.plot(df['x'], df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')
plt.show()