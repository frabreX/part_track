import numpy as np
import time
import dearpygui.dearpygui as dpg
import subprocess
import csv
import os
import threading
import json
import sys
import shutil


def rename_and_create_csv(old_path, new_path, header):
    """Rename existing CSV file and create a new one with header"""
    # Rename the old file if it exists
    if os.path.exists(old_path):
        shutil.move(old_path, new_path)
        print(f"Renamed '{old_path}' to '{new_path}'")
    else:
        print(f"No existing file at '{old_path}' to rename.")

    # Create a new CSV file with the same header
    with open(old_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        print(f"Created new CSV file at '{old_path}' with header: {header}")


def csv_stream_generator(filepath, poll_interval, max_idle_time=5):
    """Yields new rows from CSV; stops after no new data appears for a while."""
    last_position = 0
    header_skipped = False
    idle_time = 0

    while True:
        new_data_found = False

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                f.seek(last_position)
                lines = f.readlines()
                if lines:
                    new_data_found = True
                    idle_time = 0
                last_position = f.tell()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if not header_skipped:
                    header_skipped = True
                    continue
                try:
                    parts = line.split(',')
                    if len(parts) >= 6:
                        yield tuple(map(float, parts[:6]))
                except Exception as e:
                    print(f"Malformed line: {line} — {e}")

        if not new_data_found:
            idle_time += poll_interval
            if idle_time > max_idle_time:
                print("No new data for a while. Assuming simulation is finished.")
                break

        time.sleep(poll_interval)


simulation_process = None



def run_simulation():
    rename_and_create_csv("log.csv", "old_log.csv", ["step", "t", "total_time", "temp", "etot", "epart"])

    try:
        subprocess.Popen(
            [sys.executable, "MD_simulation.py"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("Simulation launched independently.")
    except Exception as e:
        dpg.set_value("status", f"Failed to start simulation: {e}")


def run_gui():
    global simulation_process
    try:
        with open("input_config.json", "r") as f:
            parameters = json.load(f)
        tmax = parameters["tmax"]
        dt = parameters["dt"]
    except Exception as e:
        dpg.set_value("status", f"Failed to load config: {e}")
        return

    dpg.set_value("status", "Simulation Running...")

    # Initialize lists to store all collected data for plotting
    all_plot_times = []
    all_plot_temps = []
    all_plot_etots = []
    all_plot_eparts = []
    all_steps = []

    # Clear existing plot data
    series_tags = ["temp_series", "etot_series", "epart_series"]
    for tag in series_tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    # Create the CSV stream generator - pass the process handle
    simulation_generator = csv_stream_generator("log.csv", 0.1)

    # Initialize variables
    series_created = False
    data_points_processed = 0
    last_update_time = time.time()
    UPDATE_INTERVAL = 0.2 # Update GUI every 0.2 seconds for more frequent updates

    try:
        for step, t, total_time, temp, etot, epart in simulation_generator:
            # Store the data
            all_plot_times.append(t)  # Use 't' for time axis
            all_plot_temps.append(temp)
            all_plot_etots.append(etot)
            all_plot_eparts.append(epart)
            all_steps.append(int(step))  # Store actual simulation step

            data_points_processed += 1

            # Create series on first iteration
            if not series_created:
                # Temperature plot
                if dpg.does_item_exist("temp_y_axis"):
                    dpg.add_line_series(list(all_plot_times), list(all_plot_temps),
                                        label="Temperature", parent="temp_y_axis", tag="temp_series")

                # Total Energy plot
                if dpg.does_item_exist("etot_y_axis"):
                    dpg.add_line_series(list(all_plot_times), list(all_plot_etots),
                                        label="Total Energy", parent="etot_y_axis", tag="etot_series")

                # Particle Energy plot
                if dpg.does_item_exist("epart_y_axis"):
                    dpg.add_line_series(list(all_plot_times), list(all_plot_eparts),
                                        label="Particle Energy", parent="epart_y_axis", tag="epart_series")

                series_created = True

            # Update plots and statistics based on time interval, not data point count
            current_time = time.time()
            if current_time - last_update_time >= UPDATE_INTERVAL:
                try:
                    # Update plot series
                    if dpg.does_item_exist("temp_series"):
                        dpg.set_value("temp_series", [list(all_plot_times), list(all_plot_temps)])
                    if dpg.does_item_exist("etot_series"):
                        dpg.set_value("etot_series", [list(all_plot_times), list(all_plot_etots)])
                    if dpg.does_item_exist("epart_series"):
                        dpg.set_value("epart_series", [list(all_plot_times), list(all_plot_eparts)])

                    # Auto-scale axes
                    axes_to_scale = ["temp_x_axis", "temp_y_axis", "etot_x_axis", "etot_y_axis",
                                     "epart_x_axis", "epart_y_axis"]
                    for axis in axes_to_scale:
                        if dpg.does_item_exist(axis):
                            dpg.fit_axis_data(axis)

                    # Update statistics
                    if len(all_plot_temps) > 0:
                        # Current values
                        if dpg.does_item_exist("current_time"):
                            dpg.set_value("current_time", f"Time: {t:.3f}")
                        if dpg.does_item_exist("current_step"):
                            dpg.set_value("current_step", f"Step: {int(step)}")
                        if dpg.does_item_exist("current_temp"):
                            dpg.set_value("current_temp", f"Temperature: {temp:.3f}")
                        if dpg.does_item_exist("current_etot"):
                            dpg.set_value("current_etot", f"Total Energy: {etot:.3f}")
                        if dpg.does_item_exist("current_epart"):
                            dpg.set_value("current_epart", f"Particle Energy: {epart:.3f}")

                        # Averages
                        if dpg.does_item_exist("avg_temp"):
                            dpg.set_value("avg_temp", f"Avg Temperature: {np.mean(all_plot_temps):.3f}")
                        if dpg.does_item_exist("avg_etot"):
                            dpg.set_value("avg_etot", f"Avg Total Energy: {np.mean(all_plot_etots):.3f}")
                        if dpg.does_item_exist("avg_epart"):
                            dpg.set_value("avg_epart", f"Avg Particle Energy: {np.mean(all_plot_eparts):.3f}")

                        # Min/Max ranges
                        temp_min, temp_max = np.min(all_plot_temps), np.max(all_plot_temps)
                        etot_min, etot_max = np.min(all_plot_etots), np.max(all_plot_etots)
                        epart_min, epart_max = np.min(all_plot_eparts), np.max(all_plot_eparts)

                        if dpg.does_item_exist("temp_range"):
                            dpg.set_value("temp_range", f"Temp Range: {temp_min:.3f} - {temp_max:.3f}")
                        if dpg.does_item_exist("etot_range"):
                            dpg.set_value("etot_range", f"Total E Range: {etot_min:.3f} - {etot_max:.3f}")
                        if dpg.does_item_exist("epart_range"):
                            dpg.set_value("epart_range", f"Part E Range: {epart_min:.3f} - {epart_max:.3f}")

                        # Progress based on time
                        progress = min(t / tmax, 1.0) if tmax > 0 else 0
                        if dpg.does_item_exist("progress_bar"):
                            dpg.set_value("progress_bar", progress)
                        if dpg.does_item_exist("data_points_processed"):
                            dpg.set_value("data_points_processed", f"Data points: {data_points_processed}")
                        if dpg.does_item_exist("simulation_progress"):
                            dpg.set_value("simulation_progress", f"Progress: {progress * 100:.1f}%")

                    last_update_time = current_time

                except Exception as gui_error:
                    print(f"GUI Update Error: {gui_error}")
                    continue

                # Allow GUI to update more frequently
                if dpg.is_dearpygui_running():
                    dpg.render_dearpygui_frame()

        # Also update GUI even when no new data (for responsiveness)
        current_time = time.time()
        if current_time - last_update_time >= UPDATE_INTERVAL:
            if dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
            last_update_time = current_time

    except Exception as e:
        dpg.set_value("status", f"Simulation Error: {e}")
        print(f"Simulation Error: {e}")

        # Wait a bit more to see if process finishes and writes final data
        print("Waiting for final data...")
        time.sleep(2.0)

        # Try to read any remaining data
        try:
            if os.path.exists("log.csv"):
                with open("log.csv", 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    final_data = list(reader)
                    if final_data:
                        print(f"Found {len(final_data)} total rows in final read")
                        # Process any remaining data
                        for row in final_data[len(all_plot_times):]:
                            if len(row) >= 6:
                                try:
                                    step, t, total_time, temp, etot, epart = map(float, row[:6])
                                    all_plot_times.append(t)
                                    all_plot_temps.append(temp)
                                    all_plot_etots.append(etot)
                                    all_plot_eparts.append(epart)
                                    all_steps.append(int(step))
                                    data_points_processed += 1
                                except ValueError:
                                    continue
        except Exception as final_read_error:
            print(f"Error in final data read: {final_read_error}")

    finally:
        # Final update of the plots and statistics
        try:
            if series_created and len(all_plot_times) > 0:
                if dpg.does_item_exist("temp_series"):
                    dpg.set_value("temp_series", [list(all_plot_times), list(all_plot_temps)])
                if dpg.does_item_exist("etot_series"):
                    dpg.set_value("etot_series", [list(all_plot_times), list(all_plot_etots)])
                if dpg.does_item_exist("epart_series"):
                    dpg.set_value("epart_series", [list(all_plot_times), list(all_plot_eparts)])

                # Final axis scaling
                axes_to_scale = ["temp_x_axis", "temp_y_axis", "etot_x_axis", "etot_y_axis",
                                 "epart_x_axis", "epart_y_axis"]
                for axis in axes_to_scale:
                    if dpg.does_item_exist(axis):
                        dpg.fit_axis_data(axis)

                # Final statistics update
                if dpg.does_item_exist("progress_bar"):
                    dpg.set_value("progress_bar", 1.0)
                if dpg.does_item_exist("data_points_processed"):
                    dpg.set_value("data_points_processed", f"Data points: {data_points_processed} (Finished)")
                if dpg.does_item_exist("simulation_progress"):
                    dpg.set_value("simulation_progress", "Progress: 100.0% (Complete)")
                if dpg.does_item_exist("current_time"):
                    dpg.set_value("current_time", f"Time: {all_plot_times[-1]:.3f}" if all_plot_times else "Time: 0.0")
                if dpg.does_item_exist("current_step"):
                    dpg.set_value("current_step", f"Step: {all_steps[-1]}" if all_steps else "Step: 0")

        except Exception as final_error:
            print(f"Final update error: {final_error}")

        # Force one final GUI update
        if dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        if dpg.does_item_exist("status"):
            dpg.set_value("status", "Simulation Finished")
        print(f"Simulation process completed. Processed {data_points_processed} data points.")

        # Print final statistics
        if all_plot_times:
            print(f"Final time: {all_plot_times[-1]:.3f}")
            print(f"Final step: {all_steps[-1] if all_steps else 'Unknown'}")
            print(f"Total data points collected: {len(all_plot_times)}")


def send_parameters():
    # Retrieve values from DPG inputs
    temp = dpg.get_value("temp")
    dt = dpg.get_value("dt")
    tmax = dpg.get_value("tmax")
    sample_interval = dpg.get_value("sample_interval")
    total_save_interval = dpg.get_value("total_save_interval")
    density = dpg.get_value("density")
    rc = dpg.get_value("rc")
    nx = dpg.get_value("nx")
    ny = dpg.get_value("ny")
    nz = dpg.get_value("nz")
    potential_type = dpg.get_value("potential")

    parameters = {
        "temp": temp,
        "dt": dt,
        "tmax": tmax,
        "sample_interval": sample_interval,
        "total_save_interval": total_save_interval,
        "density": density,
        "rc": rc,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "potential_type": potential_type,
    }

    # Salva i parametri in un file JSON
    with open("input_config.json", "w") as f:
        json.dump(parameters, f)

    dpg.set_value("parameters_status", "Parameters Saved")


# --- DearPyGui app setup ---
dpg.create_context()
dpg.create_viewport(title='MD Simulation GUI - Real-time CSV Version', width=1920, height=1080)
dpg.setup_dearpygui()

# Main control panel - left side
with dpg.window(label="Simulation Control Panel", width=400, height=1050, pos=(10, 10)):
    # Simulation Parameters Section
    dpg.add_text("Simulation Parameters", color=(255, 255, 0))
    dpg.add_separator()

    dpg.add_input_float(label="Temperature", tag="temp", default_value=1.0, width=200)
    dpg.add_input_float(label="Time Step (dt)", tag="dt", default_value=0.005, width=200)
    dpg.add_input_float(label="Max Time", tag="tmax", default_value=100.0, width=200)
    dpg.add_input_float(label="Cutoff (rc)", tag="rc", default_value=2.5, width=200)

    dpg.add_input_float(label="Density", tag="density", default_value=0.8, width=200)

    dpg.add_spacer(width=10)
    dpg.add_text("Grid Dimensions", color=(255, 255, 0))
    dpg.add_separator()

    dpg.add_input_int(label="nx", tag="nx", default_value=3, width=200)
    dpg.add_input_int(label="ny", tag="ny", default_value=3, width=200)
    dpg.add_input_int(label="nz", tag="nz", default_value=3, width=200)

    dpg.add_spacer(width=10)
    dpg.add_text("Polling interval", color=(255, 255, 0))
    dpg.add_separator()

    dpg.add_input_int(label="Sample Interval", tag="sample_interval", default_value=100, width=200)
    dpg.add_input_int(label="Total Save Interval", tag="total_save_interval", default_value=1000, width=200)

    dpg.add_spacer(width=10)
    dpg.add_text("Potential Settings", color=(255, 255, 0))
    dpg.add_separator()

    dpg.add_combo(label="Potential Type", tag="potential", items=["LJ", "WF"],
                  default_value="LJ", width=200)

    dpg.add_spacer(width=10)
    dpg.add_button(label="Save Parameters", callback=send_parameters, width=300, height=40)
    dpg.add_spacer(width=10)
    dpg.add_text("Ready to save", tag="parameters_status", color=(0, 255, 0))

    dpg.add_spacer(width=10)
    dpg.add_button(label="Run Simulation", callback=run_simulation, width=300, height=40)
    dpg.add_spacer(width=10)
    dpg.add_text("Ready", tag="status", color=(0, 255, 0))


    dpg.add_spacer(width=10)
    dpg.add_button(label="Run Control Panel", callback=run_gui, width=300, height=40)
    dpg.add_spacer(width=10)



# Temperature plot - top right (ONLY TEMPERATURE)
with dpg.window(label="Temperature Evolution", pos=(420, 10), width=750, height=340):
    with dpg.plot(label="Temperature vs Time", height=300, width=720):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="temp_x_axis")
        dpg.add_plot_legend()
        with dpg.plot_axis(dpg.mvYAxis, label="Temperature", tag="temp_y_axis"):
            pass

# Total Energy plot - middle right (ONLY TOTAL ENERGY)
with dpg.window(label="Total Energy Evolution", pos=(420, 360), width=750, height=340):
    with dpg.plot(label="Total Energy vs Time", height=300, width=720):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="etot_x_axis")
        dpg.add_plot_legend()
        with dpg.plot_axis(dpg.mvYAxis, label="Total Energy", tag="etot_y_axis"):
            pass

# Particle Energy plot - bottom right (ONLY PARTICLE ENERGY)
with dpg.window(label="Particle Energy Evolution", pos=(420, 710), width=750, height=350):
    with dpg.plot(label="Particle Energy vs Time", height=310, width=720):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="epart_x_axis")
        dpg.add_plot_legend()
        with dpg.plot_axis(dpg.mvYAxis, label="Particle Energy", tag="epart_y_axis"):
            pass

# Stats window - far right
with dpg.window(label="Statistics", pos=(1180, 10), width=730, height=1050):
    dpg.add_text("Simulation Statistics", color=(255, 255, 0))
    dpg.add_separator()

    dpg.add_text("Current Values:", color=(200, 200, 255))
    dpg.add_text("Time: 0.0", tag="current_time")
    dpg.add_text("Step: 0", tag="current_step")
    dpg.add_text("Temperature: 0.0", tag="current_temp")
    dpg.add_text("Total Energy: 0.0", tag="current_etot")
    dpg.add_text("Particle Energy: 0.0", tag="current_epart")

    dpg.add_spacer(width=10)
    dpg.add_text("Average Values:", color=(200, 200, 255))
    dpg.add_text("Avg Temperature: 0.0", tag="avg_temp")
    dpg.add_text("Avg Total Energy: 0.0", tag="avg_etot")
    dpg.add_text("Avg Particle Energy: 0.0", tag="avg_epart")

    dpg.add_spacer(width=10)
    dpg.add_text("Min/Max Values:", color=(200, 200, 255))
    dpg.add_text("Temp Range: 0.0 - 0.0", tag="temp_range")
    dpg.add_text("Total E Range: 0.0 - 0.0", tag="etot_range")
    dpg.add_text("Part E Range: 0.0 - 0.0", tag="epart_range")

    dpg.add_spacer(width=10)
    dpg.add_text("Simulation Progress:", color=(200, 200, 255))
    dpg.add_progress_bar(label="Progress", tag="progress_bar", width=400)
    dpg.add_text("Progress: 0.0%", tag="simulation_progress")
    dpg.add_text("Data points: 0", tag="data_points_processed")

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()