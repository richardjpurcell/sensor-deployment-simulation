import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

def show_fire_progression(fire_maps):
    """
    Animates the fire progression over time.
    
    :param fire_maps: List of numpy arrays representing fire maps for each time step.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def update(frame):
        ax.clear()
        ax.imshow(fire_maps[frame], cmap="YlOrRd", alpha=0.8)
        ax.set_title(f"Fire Progression - Time Step {frame}")
        ax.axis("off")
    
    ani = animation.FuncAnimation(fig, update, frames=len(fire_maps), interval=1000, repeat=True)
    plt.show()

def show_sensor_deployment(sensor_positions_history, fire_maps):
    """
    Animates the sensor deployment over the fire maps.
    
    :param sensor_positions_history: List of lists containing sensor (row, col) positions for each time step.
    :param fire_maps: List of numpy arrays representing fire maps for each time step.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def update(frame):
        ax.clear()
        ax.imshow(fire_maps[frame], cmap="YlOrRd", alpha=0.8)
        sensor_positions = sensor_positions_history[frame]
        if sensor_positions:
            sensor_positions = np.array(sensor_positions)
            ax.scatter(sensor_positions[:, 1], sensor_positions[:, 0], c="blue", s=100, marker="o")
        ax.set_title(f"Time Step {frame} - Sensor Deployment")
        ax.axis("off")
    
    ani = animation.FuncAnimation(fig, update, frames=len(fire_maps), interval=1000, repeat=True)
    plt.show()

def show_metrics(coverage_history):
    """
    Plots the coverage metric over time.
    
    :param coverage_history: List of coverage values (floats) for each time step.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(coverage_history)), coverage_history, marker="o", linestyle="-", color="green")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage Metric Over Time")
    ax.grid(True)
    ax.set_ylim(0, 1)
    plt.show()

def show_bp_maps(bp_maps):
    """
    Animates the burn probability maps over time.
    
    :param bp_maps: List of numpy arrays representing BP maps for each time step.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def update(frame):
        ax.clear()
        ax.imshow(bp_maps[frame], cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"Burn Probability Map - Time Step {frame}")
        ax.axis("off")
    
    ani = animation.FuncAnimation(fig, update, frames=len(bp_maps), interval=1000, repeat=True)
    plt.show()

def load_time_step_logs(logs_dir):
    """
    Loads all JSON log files from the specified logs directory.
    
    :param logs_dir: Directory containing time step JSON log files.
    :return: Dictionary mapping time step (int) to log data.
    """
    time_step_logs = {}
    for filename in sorted(os.listdir(logs_dir)):
        if filename.startswith("time_step_") and filename.endswith(".json"):
            filepath = os.path.join(logs_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            time_step_logs[data["time_step"]] = data
    return time_step_logs

def interactive_dashboard(fire_maps, sensor_positions_history, coverage_history, bp_maps):
    """
    Creates an interactive dashboard with three panels:
      1. Fire progression with sensor deployment (current time step).
      2. Coverage metric over time with a vertical line indicating current time.
      3. Burn probability map (current time step).
    
    Navigation is provided via a slider and Next/Previous buttons.
    
    :param fire_maps: List of numpy arrays representing fire maps per time step.
    :param sensor_positions_history: List of lists of sensor (row, col) positions per time step.
    :param coverage_history: List of coverage values (floats) for each time step.
    :param bp_maps: List of numpy arrays representing burn probability maps per time step.
    """
    num_time_steps = len(fire_maps)
    current_time = 0

    # Create a figure with three subplots side by side.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.25)

    # Panel 1: Fire progression with sensor deployment.
    ax1 = axs[0]
    fire_im = ax1.imshow(fire_maps[current_time], cmap="YlOrRd", alpha=0.8)
    sensor_scatter = None
    if sensor_positions_history[current_time]:
        sensor_positions = np.array(sensor_positions_history[current_time])
        sensor_scatter = ax1.scatter(sensor_positions[:, 1], sensor_positions[:, 0], c="blue", s=100, marker="o")
    ax1.set_title(f"Fire & Sensors - Time Step {current_time}")
    ax1.axis("off")

    # Panel 2: Coverage metric over time.
    ax2 = axs[1]
    ax2.plot(range(num_time_steps), coverage_history, marker="o", linestyle="-", color="green")
    vline = ax2.axvline(x=current_time, color="red", linestyle="--")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Coverage")
    ax2.set_title("Coverage Over Time")
    ax2.grid(True)
    ax2.set_ylim(0, 1)

    # Panel 3: Burn probability map.
    ax3 = axs[2]
    bp_im = ax3.imshow(bp_maps[current_time], cmap="hot", vmin=0, vmax=1)
    ax3.set_title(f"Burn Probability - Time Step {current_time}")
    ax3.axis("off")

    # Create a slider for time step selection.
    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    time_slider = Slider(ax_slider, 'Time Step', 0, num_time_steps - 1, valinit=current_time, valfmt='%0.0f')

    def update(val):
        nonlocal sensor_scatter
        t = int(time_slider.val)
        # Update Panel 1: Fire map and sensor positions.
        fire_im.set_data(fire_maps[t])
        if sensor_scatter is not None:
            sensor_scatter.remove()
        if sensor_positions_history[t]:
            sensor_positions = np.array(sensor_positions_history[t])
            sensor_scatter = ax1.scatter(sensor_positions[:, 1], sensor_positions[:, 0], c="blue", s=100, marker="o")
        ax1.set_title(f"Fire & Sensors - Time Step {t}")

        # Update Panel 2: Move vertical line.
        vline.set_xdata([t, t])
        ax2.relim()
        ax2.autoscale_view()

        # Update Panel 3: Burn probability map.
        bp_im.set_data(bp_maps[t])
        ax3.set_title(f"Burn Probability - Time Step {t}")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    # Create "Previous" and "Next" buttons.
    axprev = plt.axes([0.1, 0.025, 0.1, 0.04])
    axnext = plt.axes([0.8, 0.025, 0.1, 0.04])
    bprev = Button(axprev, 'Previous')
    bnext = Button(axnext, 'Next')

    def prev(event):
        t = int(time_slider.val)
        if t > 0:
            time_slider.set_val(t - 1)

    def next(event):
        t = int(time_slider.val)
        if t < num_time_steps - 1:
            time_slider.set_val(t + 1)

    bprev.on_clicked(prev)
    bnext.on_clicked(next)

    plt.show()
