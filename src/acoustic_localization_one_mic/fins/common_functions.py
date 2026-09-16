import torch


def calculate_theta(receiver_position, source_location):
    # Calculate theta
    delta_x = source_location[:, 0] - receiver_position[:, 0]
    delta_y = source_location[:, 1] - receiver_position[:, 1]
    theta = torch.atan2(delta_y, delta_x)  # Angle in radians
    # Convert radians to degrees
    theta_degrees = torch.rad2deg(theta)
    return theta_degrees


def calculate_radius(receiver_position, source_location, rounded = True):
    delta_x = source_location[:, 0] - receiver_position[:, 0]
    delta_y = source_location[:, 1] - receiver_position[:, 1]
    radius = torch.sqrt(delta_x**2 + delta_y**2)
    if rounded:
        radius_rounded = torch.round(radius * 10) / 10
    else:
        radius_rounded = radius
    return radius_rounded


def calc_rad_error(pred_radius, true_radius):
    rad_error = torch.abs(pred_radius - true_radius)
    return rad_error