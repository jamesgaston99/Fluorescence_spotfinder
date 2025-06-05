#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 13:31:23 2025

@author: jamesgaston
"""
import os
import cv2
import numpy as np
import pandas as pd
from czifile import CziFile

# === USER PARAMETERS ===
input_folder = 'your_directory'
output_excel = os.path.join(input_folder, 'single_channel_analysis_results.xlsx')

# Parameters
min_intensity = #yourthresh     # minimum intensity threshold for detecting fluorescence
area_threshold = #yourarea      # area threshold (in pixels) separating single spots vs aggregates
min_area = #yourminarea              # minimum area to consider as a valid contour
channel_to_analyze = #yourchannel    # 0 for channel 1, 1 for channel 2

# === FUNCTION: SPLIT CHANNELS ===
def splitchans(filepath, channel_index):
    with CziFile(filepath) as czi:
        img = czi.asarray()
        shape = img.shape

        if shape[1] <= channel_index:
            raise ValueError(f"Requested channel index {channel_index} out of range. Image has {shape[1]} channels.")

        channel = np.squeeze(img[0, channel_index, 0, 0, :, :, 0])
        if channel.ndim != 2:
            raise ValueError(f"Selected channel not 2D: shape {channel.shape}")

        return channel

# === FUNCTION: FIND AND CLASSIFY CONTOURS ===
def find_and_classify_contours(img, min_threshold, area_threshold, min_area):
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined_thresh = np.maximum(otsu_thresh, (img > min_threshold).astype(np.uint8) * 255)

    binary_img = combined_thresh.astype(np.uint8)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    single_spots = []
    aggregates = []
    single_mask = np.zeros_like(img, dtype=np.uint8)
    aggregate_mask = np.zeros_like(img, dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            if area <= area_threshold:
                single_spots.append(cnt)
                cv2.drawContours(single_mask, [cnt], -1, 255, -1)
            else:
                aggregates.append(cnt)
                cv2.drawContours(aggregate_mask, [cnt], -1, 255, -1)

    return single_spots, aggregates, single_mask, aggregate_mask

# === MAIN LOOP ===
results = []

for filename in os.listdir(input_folder):
    if filename.endswith('.czi'):
        filepath = os.path.join(input_folder, filename)
        print(f"Processing {filename}")
        try:
            # Load selected channel
            channel = splitchans(filepath, channel_to_analyze)
            channel_norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Classify contours
            single_spots, aggregates, single_mask, aggregate_mask = find_and_classify_contours(
                channel, min_intensity, area_threshold, min_area
            )

            # Create output image
            output_img = cv2.cvtColor(channel_norm, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(output_img, single_spots, -1, (0, 255, 255), 1)  # Yellow for single
            cv2.drawContours(output_img, aggregates, -1, (255, 0, 255), 1)   # Pink for aggregates

            out_path = os.path.join(input_folder, filename.replace('.czi', f'_ch{channel_to_analyze+1}_classified_overlay.png'))
            cv2.imwrite(out_path, output_img)

            # Record results
            results.append({
                'Filename': filename,
                'Channel_Analyzed': channel_to_analyze + 1,
                'Single_Spots': len(single_spots),
                'Aggregates': len(aggregates)
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save Excel output
df = pd.DataFrame(results)
df.to_excel(output_excel, index=False)
print("Analysis complete. Results saved to:", output_excel)
