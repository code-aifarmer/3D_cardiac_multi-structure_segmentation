import torch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

# Instantiate the metric
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

# Simulate a batch of predictions and labels
outputs = torch.rand(2, 1, 5, 5)  # Random predictions, simulating model output
labels = torch.randint(0, 2, (2, 1, 5, 5))  # Random binary labels, simulating ground truth

# Make sure outputs are in a proper format
outputs = torch.softmax(outputs, dim=1)
outputs = AsDiscrete(argmax=True)(outputs)

# Prepare inputs as list of tuples (prediction, label)
inputs = list(zip(outputs, labels))

# Update metric
try:
    dice_metric(y_pred=outputs, y=labels)
    result = dice_metric.aggregate().item()
    print("Dice Score:", result)
except AttributeError as e:
    print(f"Error: {e}\nPlease check the MONAI documentation for the correct method usage.")
except ValueError as e:
    print(f"Aggregation Error: {e}")
