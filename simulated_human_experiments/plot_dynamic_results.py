import matplotlib.pyplot as plt
import numpy as np

# Sample data
categories = ['Heuristic-Control', 'Heuristic-Interrupt', 'POMCP', 'Bayes-POMCP']
means = np.array([-54.896, -21.83, -44.78666667, -15.65333333]) + 100
std_devs = np.array([4.318173012, 10.37045804, 1.8785337072, 1.9936789])
colors = ['#FCF37F', '#CDE84D', '#45B7A1', '#3A688F']

# Create an array of x values for the categories
x = np.arange(len(categories))

# Create the bar plot
plt.bar(x, means, yerr=std_devs, capsize=5, align='center', color=colors, label='Mean', edgecolor='k')

# Add labels and title
plt.xlabel('Robot Policy')
plt.ylabel('Team Reward')
# plt.title('Bar Plot with Mean and Standard Deviation')

# Set the x-axis labels to be the category names
plt.xticks(x, categories)

# Add a legend
# plt.legend()

# Show the plot
plt.show()
# plt.savefig("frozen_lake/plots/dynamic.pdf")