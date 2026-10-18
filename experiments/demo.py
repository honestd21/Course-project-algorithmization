import matplotlib.pyplot as plt
from simulator.field import Field

field = Field.random(30, 30, seed=42)

plt.imshow(field.weed, cmap="Greens")
plt.colorbar(label="Засорение")
plt.title("Поле")
plt.show()