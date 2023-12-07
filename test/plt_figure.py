import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
fig = plt.figure(edgecolor="#f00", facecolor="#ff0", linewidth=5, figsize=(8, 3))
plt.subplot()
plt.plot(x)
plt.show()


x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]
fig = plt.figure()
fig.add_subplot(221)
plt.plot(x)
fig.add_subplot(224)
plt.plot(y)
plt.show()
