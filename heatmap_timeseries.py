from create_data import callFunction
from useful import remove_parts_of_graph_encoder_contiformer
import numpy as np
import matplotlib.pyplot as plt

def load_heatmap_timeseries():
    data = np.load("val_heatmap/data_val_heatmap.npz")
    y_spline = data['y_spline']
    y_noise_spline = data['y_noise_spline']
    min_value = data['min_value']
    max_value = data['max_value']
    noise_std = data['noise_std']
    mask = data['mask']

    y_noise_spline_masked = y_noise_spline.copy()
    y_noise_spline_masked[mask == 1] = min(y_noise_spline)
    
    x_values = np.linspace(0, 100)
    plt.plot(x_values, y_spline, label='Clean Spline')
    plt.plot(x_values, y_noise_spline_masked, label='Masked Noisy Spline')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    while True:
        # Example usage of callFunction and remove_parts_of_graph_encoder_contiformer
        x_values = np.linspace(0, 100)
        y_start = 5.0
        randomInt = 5  # Example function type
        y_spline, y_noise_spline, min_value, max_value, noise_std = callFunction(
            x_values=x_values,
            y_start=y_start,
            random_number_range=[[-1, 1], 0, 0.1],
            spline_value=[800000, 1100000],
            vocab_size=100000,
            randomInt=randomInt,
            noise_std=["norm", 0, 0.1]
        )

        plt.plot(x_values, y_spline, label='Clean Spline')
        plt.plot(x_values, y_noise_spline, label='Noisy Spline')
        plt.legend()
        plt.show()
        print(f"Noise STD: {noise_std}")

        userInput = input("Press 1 to stop timeSeries Generation")

        if(userInput == str(1)):
            break



    while True:
        mask_size = 20
        offset = 2
        mask = remove_parts_of_graph_encoder_contiformer(x_values, mask_size, offset)
        y_noise_spline_masked = y_noise_spline.copy()
        y_noise_spline_masked[mask == 1] = min(y_noise_spline)
        plt.plot(x_values, y_noise_spline_masked, label='Noise Spline Masked')
        plt.plot(x_values, y_spline, label='Clean Spline')
        plt.legend()
        plt.show()

        userInput = input("Press 1 to stop timeSeries Generation")

        if(userInput == str(1)):
            break


    #storing values in val_heatmap
    np.savez(
        "val_heatmap/data_val_heatmap.npz",
        y_spline=y_spline,
        y_noise_spline=y_noise_spline,
        min_value=min_value,
        max_value=max_value,
        noise_std=noise_std,
        mask=mask,
        mae = [],
        epoch = [],
        rmse = []
    )

    np.savez(
        "val_heatmap/heatmap_data.npz",
        mae = [],
        epoch = [],
        rmse = [],
    )

    np.savez(
        "val_heatmap/attn_data.npz",
        attn = []
    )

    load_heatmap_timeseries()



