import matplotlib.pyplot as plt
import numpy as np


def plot_static_vals(all_vals, std_vals, x_label="Expertise", y_label="Team Reward"):
    # Sample data
    categories = ['0.2', '0.4', '0.6', '0.8']

    # Plot from low to high expertise
    if x_label == "Expertise":
        values1 = all_vals[0][::-1]
        values2 = all_vals[1][::-1]
        values3 = all_vals[2][::-1]
        values4 = all_vals[3][::-1]

    else:
        values1 = all_vals[0]
        values2 = all_vals[1]
        values3 = all_vals[2]
        values4 = all_vals[3]

    # Define the width of the bars
    bar_width = 0.2

    # Set the x locations for the groups
    x = np.arange(len(categories))

    # Define a color palette
    colors = ['#FCF37F', '#CDE84D', '#45B7A1', '#3A688F']

    # Create the first set of bars
    plt.bar(x - 3 * bar_width / 2, values1, bar_width, color=colors[0], capsize=5, label='heuristic-control',
            edgecolor='k', yerr=std_vals[0])

    # Create the second set of bars
    plt.bar(x - bar_width / 2, values2, bar_width, color=colors[1], label='heuristic-interrupt', edgecolor='k',
            yerr=std_vals[1], capsize=5)

    # Create the third set of bars
    plt.bar(x + bar_width / 2, values3, bar_width, color=colors[2], label='POMCP', edgecolor='k',
            yerr=std_vals[2], capsize=5)

    # Create the fourth set of bars
    plt.bar(x + 3 * bar_width / 2, values4, bar_width, color=colors[3], label='Bayes-POMCP', edgecolor='k',
            yerr=std_vals[3], capsize=5)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim((0, 100))
    # plt.title('Four Bars for Each Category with Color Palette')
    plt.xticks(x, categories)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(f"frozen_lake/plots/{x_label}.pdf")


if __name__ == '__main__':
    # # Fixed trust, varying expertise
    expertise_mean_vals = np.array([[-50.52, -11.64, -38.93333333, -11.13333333],
                                    [-83.36, -23.88, -43.8, -7.533333333],
                                    [-121, -114.72, -45.53333333, -13.97777778],
                                    [-132.92, -150.34, -62.93333333, -27.33333333]])

    expertise_std_vals = np.array([[47.41267341, 18.35577293, 6.88444301, 2.317086293],
                                   [44.24412277, 14.54309458, 3.39803865, 4.446221867],
                                   [26.57404749, 36.48026316, 5.039400317, 5.427592969],
                                   [4.888517158, 14.62608628, 15.65488919, 1.481740718]])

    expertise_mean_vals += 100
    expertise_mean_vals = expertise_mean_vals.T

    expertise_std_vals = expertise_std_vals / np.sqrt(5)  # Standard error calculation (n=5 # Maps)
    expertise_std_vals = expertise_std_vals.T

    plot_static_vals(expertise_mean_vals, std_vals=expertise_std_vals, x_label="Expertise", y_label="Team Reward")

    # trust_mean_vals = np.array([[-73.8, -62.82, -53.46666667, -25],
    #                             [-82.4, -66, -47.46666667, -23.53333333],
    #                             [-86.16, -58.78, -43.06666667, -15.2],
    #                             [-82.62, -46.34, -37.2, -13.06666667]])
    #
    # trust_mean_vals += 100
    # trust_mean_vals = trust_mean_vals.T
    #
    # trust_std_vals = np.array([[17.45611641, 19.97692669, 12.8647667, 1.414213562],
    #                            [29.51359009, 43.80027397, 8.582669877, 6.419414996],
    #                            [34.04012926, 40.09296198, 10.23631878, 3.929376541],
    #                            [11.3042293, 49.05220077, 2.355136231, 5.881798667]])
    #
    # trust_std_vals = trust_std_vals / np.sqrt(5)
    # trust_std_vals = trust_std_vals.T
    #
    # plot_static_vals(trust_mean_vals, std_vals=trust_std_vals, x_label="Compliance", y_label="Team Reward")
