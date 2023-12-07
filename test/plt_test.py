import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([0, 1, 2, 36])
ypoints = np.array([0, 20, 50, 100])

# plt.plot(losses, label="Training Loss")
plt.scatter(xpoints, ypoints, color="red", label="red")
plt.plot(xpoints, ypoints, color="green", label="green")
plt.xlabel("xxxx")
plt.ylabel("yyyy")
plt.title("title")
plt.legend()

plt.show()


# xpoints2 = np.array([1, 2, 6, 8])
# ypoints2 = np.array([3, 8, 1, 10])

# plt.plot(xpoints2, ypoints2)
# plt.show()
