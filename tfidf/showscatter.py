import numpy as np
import matplotlib.pyplot as plt
import common

# Show scatter plot by matplotlib
y_test = np.loadtxt(common.FILE_TEST_SCORE)
y_train = np.loadtxt(common.FILE_TRAIN_SCORE)
s_test = np.loadtxt(common.FILE_TEST_RESULT)
s_train = np.loadtxt(common.FILE_TRAIN_RESULT)

poly_coeff = np.loadtxt(common.FILE_POLY_COEFF)


poly_enabled = False
rows = 2
if poly_coeff.shape[0] != 0:
    poly_enabled = True
    rows += 1

row = 1

plt.subplot(1, rows, row)
plt.scatter(y_test, s_test)
plt.title('Actual Vs Predicted: Test Data')
plt.xlabel('Actual')
plt.ylabel('Predicted')

row += 1

plt.subplot(1, rows, row)
plt.scatter(y_train, s_train)
plt.title('Actual Vs Predicted: Train Data')
plt.xlabel('Actual')
plt.ylabel('Predicted')

row += 1

if poly_enabled:
    plt.subplot(1, rows, row)
    x_poly = np.arange(100) / 20
    y_poly = np.polynomial.polynomial.polyval(x_poly, poly_coeff)
    plt.plot(x_poly, y_poly)
    plt.xticks(np.arange(6))
    plt.yticks(np.arange(6))

plt.show()