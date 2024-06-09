import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Generate some random data
np.random.seed(0)
data = pd.DataFrame({
    'x': np.random.rand(50),
    'y': np.random.rand(50),
    'z': np.random.rand(50) * 1000,
})

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter('x', 'y', s='z', data=data, alpha=0.6)
plt.title('Random Data Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Z-axis')
plt.grid(True)
plt.show()
