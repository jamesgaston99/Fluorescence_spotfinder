#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:27:54 2023

@author: jamesgaston
"""

import os
import cv2
import numpy as np
import czifile
import pandas as pd
import matplotlib.pyplot as plt


# Function to calculate and save histograms
def calculate_and_save_histogram(data, title, x_label, output_filename):
    hist, bins = np.histogram(data, bins=100, range=(0, 15000))

    # Create a histogram plot
    plt.figure()
    plt.hist(data, bins=100, range=(0, 15000), density=False)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Pixel counts')

    # Save the histogram as a PNG file
    plt.savefig(output_filename)
    plt.close()

# Function to plot individual contours
def plot_individual_contours(contours_data, output_filename):
    plt.figure()
    for data, color, label in contours_data:
        areas = data["Contour Area"]
        pixel_values = data["Pixel Values"]
        plt.scatter(areas, pixel_values, color=color, label=label, alpha=0.5)
    plt.title('Individual Contours: Area vs Pixel Values')
    plt.xlabel('Contour Area')
    plt.ylabel('Mean Pixel Values')
   # plt.legend()

    # Save the scatter plot as a PNG file
    plt.savefig(output_filename)
    plt.close()


# Function to process an individual image
def process_image(image_path, channel_index, threshold_value, area_threshold, all_aggregate_data, all_small_spot_data):
    try:
        # Open the CZI file
        czi_file = czifile.CziFile(image_path)

        # Read the CZI file and extract the data from the specified channel
        image = czi_file.asarray()
        channel_data = image[0, channel_index, 0, 0, :, :]

        # Convert uint16 array to binary array
        binary_array = (channel_data > threshold_value).astype(np.uint8) * 255

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize lists to store contour information
        aggregate_contours = []
        small_spot_contours = []
        aggregate_data = {"Contour Area": [], "Pixel Values": []}
        small_spot_data = {"Contour Area": [], "Pixel Values": []}

        # Iterate through the contours and classify them based on the area threshold
        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= area_threshold:
                aggregate_contours.append(contour)
                average_pixel_value_aggregate = np.mean(channel_data[contour[:, 0, 1], contour[:, 0, 0]])
                aggregate_data["Contour Area"].append(area)
                aggregate_data["Pixel Values"].append(average_pixel_value_aggregate)
            elif area < area_threshold:
                small_spot_contours.append(contour)
                average_pixel_value_small_spot = np.mean(channel_data[contour[:, 0, 1], contour[:, 0, 0]])
                small_spot_data["Contour Area"].append(area)
                small_spot_data["Pixel Values"].append(average_pixel_value_small_spot)


        # Create a black image to draw the selected contours
        result_image = np.zeros_like(binary_array, dtype=np.uint8)

        # Draw the selected contours on the result image
        result_image = np.zeros_like(binary_array, dtype=np.uint8)
        cv2.drawContours(result_image, small_spot_contours, -1, 128, 2)
        cv2.drawContours(result_image, aggregate_contours, -1, 255, 2)

        # Determine the average pixel value within the masked region in the binary image
        average_pixel_value_aggregate = np.mean(channel_data[result_image == 255])
        average_pixel_value_small_spot = np.mean(channel_data[result_image == 128])

        # Extract the filename from the full filepath
        image_name = os.path.basename(image_path)

        # Create a dictionary to store the results
        result_dict = {
            "Image": image_name,
            "Number of aggregate contours": len(aggregate_contours),
            "Number of small spot contours": len(small_spot_contours),
            "Average area of aggregate contours": np.mean([cv2.contourArea(c) for c in aggregate_contours]),
            "Max area of aggregate contours": np.max([cv2.contourArea(c) for c in aggregate_contours]),
            #"Total area of aggregate contours": sum([cv2.contourArea(c) for c in aggregate_contours]),
            "Average Pixel Value within Aggregate Contours": average_pixel_value_aggregate,
            "Average area of small spot contours": np.mean([cv2.contourArea(c) for c in small_spot_contours]),
            "Max area of small spot contours": np.max([cv2.contourArea(c) for c in small_spot_contours]),
            #"Total area of small spot contours": sum([cv2.contourArea(c) for c in single_molecule_contours]),
            "Average Pixel Value within small spot Contours": average_pixel_value_small_spot,
            "Total Pixels Above Threshold": np.sum(channel_data > threshold_value),
            "Total Pixels Below Threshold": np.sum(channel_data <= threshold_value),
            "Average pixel value above threshold": np.mean(channel_data[channel_data > threshold_value]),
            "Average pixel value below threshold": np.mean(channel_data[channel_data <= threshold_value])
            
            
        }
 
        # Calculate and save histograms for pixel brightness
        calculate_and_save_histogram(channel_data[result_image == 255], "Aggregate Brightness Histogram", "Brightness", "yourdirectory/aggregate_brightness_histogram.png")
        calculate_and_save_histogram(channel_data[result_image == 128], "Small spot Brightness Histogram", "Brightness", "yourdirectory/small spot_brightness_histogram.png")
        
        
        # Append the data to the global lists
        all_aggregate_data.append(aggregate_data)
        all_small_spot_data.append(small_spot_data)
        
        # Calculate and save histograms for pixel areas
       # calculate_and_save_histogram([cv2.contourArea(c) for c in aggregate_contours], "Aggregate Area Histogram", "Area", "/Users/jamesgaston/Desktop/PhD/Lab work/PEG blocking/PEG blocking code reanalysis/results_221020/aggregate_area_histogram.png")
       # calculate_and_save_histogram([cv2.contourArea(c) for c in small_spot_contours], "Single Molecule Area Histogram", "Area", "/Users/jamesgaston/Desktop/PhD/Lab work/PEG blocking/PEG blocking code reanalysis/results_221020/single_molecule_area_histogram.png")

        return result_dict

    except Exception as e:
        print(f"An error occurred while processing {image_path}: {str(e)}")
        return None

# Directory containing images
image_directory = 'yourdirectory'

# Parameters
channel_index = #imagechannel
threshold_value = #yourthresh
area_threshold = #yourareathresh

# List image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(".czi")]

# Initialize a list to store the results
results = []


# Initialize lists to store contour data across all images
all_aggregate_data = []
all_small_spot_data = []

# Process each image in the directory and store the results
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    result = process_image(image_path, channel_index, threshold_value, area_threshold, all_aggregate_data, all_small_spot_data)
    if result:
        results.append(result)

# Plot and save individual contours across all images
contours_data = [
    (data, 'green', 'Aggregate Contours') for data in all_aggregate_data
] + [
    (data, 'blue', 'Small Spot Contours') for data in all_small_spot_data
]
plot_individual_contours(contours_data, "yourdirectory/individual_contours_plot.png")


# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the results to an Excel file
output_excel_file = 'yourdirectory/results.xlsx'
df.to_excel(output_excel_file, index=False, engine='openpyxl')

print(f"Results saved to {output_excel_file}")
