import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Sample data to simulate the user's dataset
# Normally, we would use the provided dataset like this:
# data = pd.read_csv("part5-unsupervised-learning/customer_data.csv")
data = pd.read_csv("a6-predictive-models-iespadas/part5-unsupervised-learning/customer_data.csv")

x = data[["Annual Income", "Spending Score"]]

# Standardize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# The value of k has been defined
k = 5

# Apply the KMeans algorithm
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(x_scaled)

# Get the centroid and label values
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Set the size of the graph
plt.figure(figsize=(5, 4))

# Use a for loop to plot the data points in each cluster
colors = ['blue', 'green', 'red', 'cyan', 'magenta']
for i in range(k):
    # Separate data points by cluster
    cluster_data = x_scaled[labels == i]
    # Plot the data points
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, c=colors[i], label=f'Cluster {i+1}')

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='yellow', label='Centroids')

# Show the graph with labels
plt.xlabel("Annual Income (standardized)")
plt.ylabel("Spending Score (standardized)")
plt.title("Customer Segments")
plt.legend()
plt.show()
