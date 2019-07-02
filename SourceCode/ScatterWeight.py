import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


#y = np.arange(N)
r1 = np.loadtxt("E:\Study_materials\Semester-4\Project\SourceCode\Mid_Sem_Code\FashionMaster.01_backup\FashionMaster.01_backup\PlotData\Re1.txt")
#r2 = np.loadtxt("E:\Study_materials\Semester-4\Project\SourceCode\Mid_Sem_Code\FashionMaster.01_backup\FashionMaster.01_backup\PlotData\Re2.txt")

N = r1.size
x = np.arange(N)


#y = np.random.rand(N)
colors = np.random.rand(N)
#area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
area = (5)**2  # 0 to 15 point radii

plt.scatter(x, r1, s=area, c=colors, alpha=0.5)

plt.title('Input Data to the RELU in the first intermidiate Layer')
plt.ylabel('Input Data')
plt.xlabel('Neuron in Intermidiate Layer')
plt.show()