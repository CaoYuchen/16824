import numpy as np
import matplotlib.pyplot as plt

y = [100, 300, 200, 500, 1000]
x = [2, 3, 1, 4, 5]

plt.bar(x, y, width=0.5, align="center")
# y_offset = y_offset + data[row]
# cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])

plt.ylabel("test")
# plt.yticks(values * value_increment, ['%d' % val for val in values])
# plt.xticks([])
plt.title('Loss by Disaster')

plt.show()
