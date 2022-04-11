import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import geometricbrownianmotion as gbm
import auxmodels
import numpy as np


model = tf.keras.models.load_model("ml_model")
model.summary()
real_stock = pd.read_csv("E-MiniSPClosingPrice.txt").to_numpy()
params = auxmodels.maximum_likelihood_naive(real_stock)
pred = model.predict(params.T)
mu, sigma = pred[0]
print(mu, sigma)
ts = gbm.gbm_exact(mu, sigma, real_stock[-1], 1, 15, 10000)
x = np.arange(real_stock.size - 1, real_stock.size - 1 + np.shape(ts)[0])
plt.figure()
plt.plot(x, ts)
plt.plot(real_stock, linewidth=2, color="k")
plt.xlim(240, real_stock.size + 10)

plt.figure()
plt.hist(ts[1, :], bins="auto")
plt.show()

