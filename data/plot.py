from copy import copy

import matplotlib.pyplot as plt


#
# 100%: 95.11%, 95.24%
# 80%: 93.98, 95.37
# 60%: 92.89, 95.54
# 40%: 0.9414, 0.9509
# 20%: 0.8739, 0.9349
# 10%: 0.7945, 0.8932
# 1%: 0.6296, 0.8519

x_1 = [1, 2, 3, 9]
x_2 = copy(x_1)
x_3 = copy(x_1)

y_act = [0.9547, 0.9523, 0.9378, 0.9250]
y_gender = [0.9541, 0.9465, 0.9471, 0.9146]
y_char = [act / gender for (act, gender) in zip (y_act, y_gender)]

plt.plot(x_1, y_act, label = "Activity")
plt.plot(x_2, y_gender, label = "Gender")
plt.plot(x_3, y_char, label = "Characterization")
plt.xlabel('Gausion noise Strength')
plt.ylabel('Classification Accuracy')
plt.title('Model Performance under Different Noise Strength for Training Data')
plt.legend()
plt.show()
