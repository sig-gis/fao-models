import matplotlib.pyplot as plt

def plot_training_curves(dictionaries, keys_to_plot):
    line_styles = ['-', '--', '-.', ':']  # List of line styles
    colors = ['blue', 'green', 'red', 'purple']  # List of colors

    # Iterate over each dictionary and key
    for i, dictionary in enumerate(dictionaries):
        for j, key_to_plot in enumerate(keys_to_plot):
            # Get the values for the specified key
            values = dictionary[key_to_plot]

            # Create the line graph with different line styles and colors
            plt.plot(values, linestyle=line_styles[j % len(line_styles)], color=colors[i % len(colors)])

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curves - {}'.format(', '.join(keys_to_plot)))
    
    # Create a list of legend items
    legend_items = []
    for dictionary in dictionaries:
        for key_to_plot in keys_to_plot:
            legend_items.append(dictionary['experiment_name'])
    
    plt.legend(legend_items)  # Display legend items
    plt.show()