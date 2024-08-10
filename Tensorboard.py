from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

# 设置日志目录
logdir = 'C:/Users/12234/Desktop/log.0'

from tensorboard.backend.event_processing import event_accumulator

# Path to the directory where the TensorBoard logs are stored
log_path = 'C:/Users/12234/Desktop/runs'

ea = event_accumulator.EventAccumulator(log_path,
    size_guidance={event_accumulator.IMAGES: 10})  # Adjust based on how many items you want to load

ea.Reload()  # Load the events from disk
print(ea.Tags())
# Assuming 'images' is the correct tag name
image_events = ea.Images('images')

for image_event in image_events:
    # Process your image_event
    print(image_event)  # Example processing
