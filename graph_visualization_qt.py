# --- START OF FILE graph_visualization_qt.py ---
import matplotlib.pyplot as plt
import numpy as np
import logging
import io      
import cv2 # Only needed if get_frame_as_array uses cv2.imdecode, which it will

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphVisualizerQt: # Renamed to avoid conflict if old one is in same dir
    def __init__(self, processor):
        self.processor = processor
        self.figure = None # Will be set by MainAppWindow
        self.ax = None     # Will be set by MainAppWindow
        self.lines = {} 
        self.full_data_cache = {}
        self.active_legend = None
        self.default_dpi = 100 
        self.current_time_indicator_on_graph = None # Store ref to the axvline on graph

    def set_figure_axes(self, fig, ax):
        """Called by the main Qt app to provide the drawing context."""
        self.figure = fig
        self.ax = ax
        # Initial setup of axes properties if needed, or rely on plot_tooth_lines
        if self.ax:
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Average Force (N)")
            self.ax.set_title("Average Bite Force Over Time")
            self.ax.grid(True)
            if self.processor.timestamps and len(self.processor.timestamps) > 0:
                self.ax.set_xlim(self.processor.timestamps[0], self.processor.timestamps[-1])
            else: 
                self.ax.set_xlim(0, 1) 
            # Ylim will be set more dynamically in plot_tooth_lines or update

    def plot_tooth_lines(self, tooth_ids_to_display):
        if self.ax is None:
            logging.error("GraphVisualizerQt: Axes not set. Cannot plot lines.")
            return

        # Clear previous lines and legend
        for line_artist in self.ax.lines: line_artist.remove() # More robust clearing
        self.lines.clear(); self.full_data_cache.clear()
        if self.active_legend:
            try: self.active_legend.remove()
            except AttributeError: pass # Might already be gone if axes were cleared
            self.active_legend = None

        if not tooth_ids_to_display:
            title_suffix = "(No tooth selected)"
            self.ax.set_title(f"Average Bite Force Over Time {title_suffix}")
            if self.figure: self.figure.canvas.draw_idle()
            return

        # Determine Y-axis limits based on the data of teeth to be displayed
        max_y_force_for_current_plot = 10.0 # Default minimum
        temp_forces_for_ylim = []
        for tooth_id in tooth_ids_to_display:
            _ , forces = self.processor.get_average_force_for_tooth(tooth_id)
            if forces.size > 0:
                temp_forces_for_ylim.append(forces)
        
        if temp_forces_for_ylim:
            max_y_force_for_current_plot = np.max(np.concatenate(temp_forces_for_ylim))
        max_y_force_for_current_plot = max(max_y_force_for_current_plot, 10.0) # Ensure at least 10
        self.ax.set_ylim(bottom=-max_y_force_for_current_plot*0.05, top=max_y_force_for_current_plot * 1.1)


        num_lines = len(tooth_ids_to_display)
        colors = plt.cm.viridis(np.linspace(0,1,max(1,num_lines)))

        for i, tooth_id in enumerate(tooth_ids_to_display):
            full_times, full_forces = self.processor.get_average_force_for_tooth(tooth_id)
            self.full_data_cache[tooth_id] = (full_times, full_forces)
            line, = self.ax.plot([], [], label=f"Tooth {tooth_id}", color=colors[i % len(colors)], lw=1.5)
            self.lines[tooth_id] = line
        
        self.active_legend = self.ax.legend(loc='upper right')
        title_suffix = f"(Teeth: {', '.join(map(str, tooth_ids_to_display))})"
        self.ax.set_title(f"Average Bite Force Over Time {title_suffix}")
        
        logging.info("Graph lines plotted for teeth: %s", tooth_ids_to_display)
        if self.figure: self.figure.canvas.draw_idle()

    def update_graph_to_timestamp(self, current_timestamp, tooth_ids_currently_plotted):
        if self.figure is None or self.ax is None: return

        for tooth_id in tooth_ids_currently_plotted: 
            if tooth_id in self.lines and tooth_id in self.full_data_cache:
                full_times, full_forces = self.full_data_cache[tooth_id]
                if full_times is not None and len(full_times) > 0:
                    idx_up_to_time = np.searchsorted(full_times, current_timestamp, side='right')
                    times_to_plot = full_times[:idx_up_to_time]
                    forces_to_plot = full_forces[:idx_up_to_time]
                    self.lines[tooth_id].set_data(times_to_plot, forces_to_plot)
                else: self.lines[tooth_id].set_data([], []) 
        # The main AnimationController will call figure.canvas.draw_idle() for the live view.

    def update_time_indicator(self, current_timestamp):
        """Updates or creates the vertical time indicator line."""
        if not self.ax or not self.figure: return

        if self.current_time_indicator_on_graph:
            try: self.current_time_indicator_on_graph.remove()
            except (ValueError, AttributeError): pass # Already removed or None
            self.current_time_indicator_on_graph = None
        
        if current_timestamp is not None:
            self.current_time_indicator_on_graph = self.ax.axvline(
                current_timestamp, color='grey', linestyle=':', lw=1, gid="graph_time_indicator_live"
            )
        # Figure redraw will be handled by the main animation loop via draw_idle() on canvas

    def get_frame_as_array(self, current_timestamp, tooth_ids_to_display):
        if self.figure is None or self.ax is None:
            logging.warning("Graph figure not initialized for get_frame_as_array.")
            return None # Cannot generate frame

        # Ensure graph is updated to the specific timestamp for the screenshot
        # This might redraw lines if tooth_ids_to_display changed from current state
        if set(tooth_ids_to_display) != set(self.lines.keys()):
            self.plot_tooth_lines(tooth_ids_to_display) 
            
        self.update_graph_to_timestamp(current_timestamp, tooth_ids_to_display)
        self.update_time_indicator(current_timestamp) # Ensure indicator is on for screenshot
        
        self.figure.canvas.draw() # Ensure all drawing commands are processed
        
        buf = io.BytesIO()
        try:
            self.figure.savefig(buf, format='png', dpi=self.default_dpi, bbox_inches='tight', facecolor=self.figure.get_facecolor())
            buf.seek(0)
            img_array_png = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            frame_bgr = cv2.imdecode(img_array_png, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"Error saving Matplotlib figure to buffer: {e}")
            frame_bgr = None
        finally:
            buf.close()
            if self.current_time_indicator_on_graph: # Clean up indicator used for screenshot
                try: self.current_time_indicator_on_graph.remove()
                except (ValueError, AttributeError): pass
                self.current_time_indicator_on_graph = None

        if frame_bgr is None: logging.error("Failed to decode Matplotlib figure to image array.")
        return frame_bgr
# --- END OF FILE graph_visualization_qt.py ---