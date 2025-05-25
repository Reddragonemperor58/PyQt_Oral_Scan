# --- START OF FILE main_qt_app.py ---
import sys
import logging
import numpy as np
import cv2 # For video writing and image manipulation
import atexit
import os
# --- CHOOSE YOUR QT BINDING ---
# This block attempts to use PySide6, then falls back to PyQt5
QT_BINDING_CHOICE = "PyQt5" # Or "PySide6" 
# QT_BINDING_CHOICE = "PySide6" 

if QT_BINDING_CHOICE == "PySide6":
    try:
        from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSplitter
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtOpenGLWidgets import QOpenGLWidget 
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        QT_BINDING = "PySide6"
        logging.info("Using PySide6.")
    except ImportError as e_pyside:
        logging.warning(f"PySide6 not found ({e_pyside}), trying PyQt5.")
        QT_BINDING_CHOICE = "PyQt5" # Fallback to PyQt5

if QT_BINDING_CHOICE == "PyQt5":
    try:
        from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSplitter
        from PyQt5.QtCore import QTimer, Qt
        from PyQt5.QtOpenGL import QGLWidget # QGLWidget is often used with QVTKRenderWindowInteractor in PyQt5
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        QT_BINDING = "PyQt5"
        logging.info("Using PyQt5.")
    except ImportError as e_pyqt5:
        logging.critical(f"Neither PySide6 nor PyQt5 found ({e_pyqt5}). Application cannot run.")
        sys.exit("PyQt5 or PySide6 is required.")


from data_acquisition import SensorDataReader
from data_processing import DataProcessor
from graph_visualization_qt import GraphVisualizerQt 
from dental_arch_grid_visualization_qt import DentalArchGridVisualizerQt
from dental_arch_3d_bar_visualization_qt import DentalArch3DBarVisualizerQt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from vedo import Plotter, settings # Import base Plotter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_main_app_window_instance_for_atexit = None

def cleanup_on_exit():
    global _main_app_window_instance_for_atexit
    if _main_app_window_instance_for_atexit and hasattr(_main_app_window_instance_for_atexit, 'video_writer'):
        if _main_app_window_instance_for_atexit.video_writer is not None and \
           _main_app_window_instance_for_atexit.video_writer.isOpened():
            logging.info("ATEIXT: Releasing OpenCV video writer...")
            _main_app_window_instance_for_atexit.video_writer.release()
            _main_app_window_instance_for_atexit.video_writer = None
            logging.info("ATEIXT: OpenCV video writer released.")
atexit.register(cleanup_on_exit)


class VedoQtCanvas(QVTKRenderWindowInteractor):
    """A QWidget that embeds a VTK Render Window for Vedo."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Configure interactor style or other VTK-level settings if needed here
        # e.g., self.setFocusPolicy(Qt.StrongFocus)

    def GetPlotter(self, **kwargs_for_plotter):
        """Creates and returns a Vedo Plotter associated with this widget's render window."""
        # Pass this QVTK widget to the plotter using qt_widget
        plt = Plotter(qt_widget=self, **kwargs_for_plotter) # Pass kwargs here
        return plt
    
    def closeEvent(self, event):
        self.Finalize() # Ensure VTK interactor is cleaned up
        super().closeEvent(event)


class EmbeddedVedoWidget(QWidget): # Wrapper for VedoQtCanvas
    def __init__(self, processor_instance, VisualizerClass, parent=None, plotter_kwargs=None):
        super().__init__(parent)
        if plotter_kwargs is None: plotter_kwargs = {}
        self.vlayout = QVBoxLayout(self); self.vlayout.setContentsMargins(0,0,0,0)
        self.vedo_canvas = VedoQtCanvas(self) # This is QVTKRenderWindowInteractor
        self.vlayout.addWidget(self.vedo_canvas)
        self.plt = self.vedo_canvas.GetPlotter(**plotter_kwargs) # Get the plotter
        
        if not self.plt:
            logging.error(f"Failed to create Plotter for {VisualizerClass.__name__}.")
            self.visualizer = None; return

        self.visualizer = VisualizerClass(processor_instance, self.plt)
        
        # Initial static setup by the visualizer
        if isinstance(self.visualizer, DentalArchGridVisualizerQt):
            if hasattr(self.visualizer, '_initialize_static_grid_elements'): self.visualizer._initialize_static_grid_elements()
            if hasattr(self.visualizer, '_fit_camera_to_grid'): self.visualizer._fit_camera_to_grid() 
        elif isinstance(self.visualizer, DentalArch3DBarVisualizerQt):
            if hasattr(self.visualizer, '_initialize_static_elements'): self.visualizer._initialize_static_elements()
            # Example: if hasattr(self.visualizer, '_fit_camera_3d'): self.visualizer._fit_camera_3d()

        if hasattr(self.visualizer, 'tooth_cell_definitions') and self.visualizer.tooth_cell_definitions:
            if hasattr(self.visualizer.processor, 'calculate_cof_trajectory'):
                 self.visualizer.processor.calculate_cof_trajectory(self.visualizer.tooth_cell_definitions)
        
        # IMPORTANT: Initial render of the Vedo canvas after setup
        # This helps establish the VTK pipeline within the Qt widget.
        self.vedo_canvas.Render()

    def update_view(self, timestamp):
        if self.visualizer and hasattr(self.visualizer, 'animate') and self.visualizer.timestamps:
            # ... (timestamp syncing logic for self.visualizer.current_timestamp_idx) ...
            try:
                if len(self.visualizer.timestamps) > 0:
                    idx = np.argmin(np.abs(np.array(self.visualizer.timestamps) - timestamp))
                    self.visualizer.current_timestamp_idx = idx
                else: return
            except Exception: return 

            self.visualizer.animate() # This calls visualizer's render_arch/display (which now only updates actors)
            
            # Now, explicitly tell the QVTK canvas (self.vedo_canvas) to render the updated scene
            if hasattr(self.vedo_canvas, 'Render') and self.vedo_canvas.isVisible():
                 self.vedo_canvas.Render() 
            # Fallback if Render method is not directly on vedo_canvas, but on its internal plotter
            elif self.plt and self.plt.renderer and self.plt.window:
                 self.plt.render() # self.plt is the plotter associated with vedo_canvas
                 

    def get_frame_as_array(self, timestamp):
        # ... (get_frame_as_array as before, ensure it calls update_view first to get correct state) ...
        if self.visualizer and hasattr(self.visualizer, 'get_frame_as_array'):
            # The visualizer's get_frame_as_array should handle its own rendering for the specific timestamp
            return self.visualizer.get_frame_as_array(timestamp)
        elif self.plt: 
            logging.warning(f"Visualizer for {getattr(self.plt,'title','Unknown Plotter')} missing get_frame_as_array. Using direct screenshot.")
            # Ensure this plotter is up-to-date for the screenshot
            if self.visualizer and hasattr(self.visualizer, 'animate') and self.visualizer.timestamps:
                original_idx = self.visualizer.current_timestamp_idx
                try:
                    if len(self.visualizer.timestamps) > 0:
                        idx = np.argmin(np.abs(np.array(self.visualizer.timestamps) - timestamp))
                        self.visualizer.current_timestamp_idx = idx
                        self.visualizer.animate() # This should call render_arch/display -> self.plt.render()
                except Exception: pass
                self.visualizer.current_timestamp_idx = original_idx 
            
            # After visualizer updates its plotter (self.plt), render the canvas before screenshot
            if hasattr(self.vedo_canvas, 'Render') and self.vedo_canvas.isVisible():
                self.vedo_canvas.Render()
            return self.plt.screenshot(asarray=True)
        return None

    def Render(self): # Expose Render method of the canvas
        if hasattr(self.vedo_canvas, 'Render'): self.vedo_canvas.Render()


class MatplotlibCanvas(FigureCanvas): # ... (same as before) ...
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi); self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig); self.setParent(parent)

class MainAppWindow(QMainWindow):
    def __init__(self, processor):
        # ... (initializations as before) ...
        super().__init__(); self.processor = processor; self.current_timestamp_idx = 0
        self.animation_timer = QTimer(self); self.is_animating = False; self.graph_time_indicator = None
        self.setWindowTitle("Dental Force Visualization Suite (PyQt)"); self.setGeometry(50, 50, 1800, 960) 
        self.initial_graph_teeth = []; self.currently_graphed_tooth_ids = []; self.last_animated_timestamp = None
        self.output_video_filename="composite_dental_animation.mp4"; self.canvas_width=1920; self.canvas_height=1080
        self.fps = 10; self.video_writer = None 
        global _main_app_window_instance_for_atexit; _main_app_window_instance_for_atexit = self
        # 1. Matplotlib Graph
        self.graph_qt_canvas = MatplotlibCanvas(self, width=16, height=4, dpi=90) # Adjusted height/dpi
        self.graph_visualizer = GraphVisualizerQt(self.processor)
        self.graph_visualizer.set_figure_axes(self.graph_qt_canvas.fig, self.graph_qt_canvas.axes)
        if self.processor.tooth_ids:
            self.initial_graph_teeth=[self.processor.tooth_ids[0],self.processor.tooth_ids[1]] if len(self.processor.tooth_ids)>=2 else self.processor.tooth_ids[:1]
            self.currently_graphed_tooth_ids = list(self.initial_graph_teeth) 
            if self.initial_graph_teeth : self.graph_visualizer.plot_tooth_lines(self.initial_graph_teeth)

        # 2. Vedo 2D Grid Visualizer (Left Vedo Panel)
        self.vedo_grid_widget = EmbeddedVedoWidget(self.processor, DentalArchGridVisualizerQt, self, plotter_kwargs={'axes': 0, 'title': "2D Grid View"})
        if hasattr(self.vedo_grid_widget.visualizer, 'set_animation_controller_for_graph_link'):
            self.vedo_grid_widget.visualizer.set_animation_controller_for_graph_link(self)

        # 3. Vedo 3D Bar Visualizer (Right Vedo Panel)
        self.vedo_3d_bar_widget = EmbeddedVedoWidget(self.processor, DentalArch3DBarVisualizerQt, self, plotter_kwargs={'axes': 0, 'title': "3D Bar View"})
        
        # 4. Detailed Info Label
        self.detailed_info_label = QLabel("Click on a tooth in the grid to see details.")
        self.detailed_info_label.setWordWrap(True)
        self.detailed_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detailed_info_label.setMinimumWidth(200) # Give it a reasonable minimum width
        self.detailed_info_label.setMaximumWidth(300) # And a maximum

        self._setup_ui()
        self._setup_animation_timer()
        
        if self.processor.timestamps:
            first_ts = self.processor.timestamps[0]; self.last_animated_timestamp = first_ts
            self.vedo_grid_widget.update_view(first_ts); self.vedo_3d_bar_widget.update_view(first_ts)

    def _initialize_video_writer(self):
        global _main_app_window_instance_for_atexit # Renamed from _video_writer_for_atexit
        if self.video_writer is None:
            if os.path.exists(self.output_video_filename): # Attempt to remove old file
                try: os.remove(self.output_video_filename); logging.info(f"Removed existing: {self.output_video_filename}")
                except Exception as e: logging.warning(f"Could not remove {self.output_video_filename}: {e}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
            self.video_writer = cv2.VideoWriter(
                self.output_video_filename, fourcc, float(self.fps), 
                (self.canvas_width, self.canvas_height)
            )
            _main_app_window_instance_for_atexit = self # Store the instance for atexit
            if not self.video_writer.isOpened():
                logging.error(f"Could not open video writer for {self.output_video_filename}")
                self.video_writer = None
                _main_app_window_instance_for_atexit = None # Clear if failed
            else: 
                logging.info(f"Video writer opened for {self.output_video_filename} at {self.fps} FPS.")
        return self.video_writer is not None and self.video_writer.isOpened()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_vertical_layout = QVBoxLayout(central_widget) # Main layout for the window

        # --- Top Area Splitter (Horizontal: Grid | 3D Bar | Info) ---
        top_horizontal_splitter = QSplitter(Qt.Horizontal) # Widgets arranged horizontally

        top_horizontal_splitter.addWidget(self.vedo_grid_widget) 
        top_horizontal_splitter.addWidget(self.vedo_3d_bar_widget)
        top_horizontal_splitter.addWidget(self.detailed_info_label)

        top_horizontal_splitter.setSizes([350, 350, 150]) # Adjust initial proportions

        # --- Main Vertical Splitter (Top Area | Graph Area) ---
        main_vertical_splitter = QSplitter(Qt.Vertical)
        
        # --- CORRECTED: Add the top_horizontal_splitter (which is a QWidget) directly ---
        main_vertical_splitter.addWidget(top_horizontal_splitter) # top_horizontal_splitter IS the container
        # --- END CORRECTION ---

        main_vertical_splitter.addWidget(self.graph_qt_canvas) # Add Matplotlib graph canvas

        main_vertical_splitter.setSizes([650, 250]) # Adjusted proportions

        main_vertical_layout.addWidget(main_vertical_splitter)

        # --- Controls Area ---
        controls_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play/Pause")
        self.play_pause_button.clicked.connect(self.toggle_animation)
        
        self.reset_3d_view_button = QPushButton("Reset 3D View")
        self.reset_3d_view_button.clicked.connect(self.reset_3d_bar_camera)

        controls_layout.addStretch(1) 
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.reset_3d_view_button)
        controls_layout.addStretch(1)
        main_vertical_layout.addLayout(controls_layout)



    def _setup_animation_timer(self): self.animation_timer.timeout.connect(self.animation_step)
    def toggle_animation(self):
        if self.is_animating:
            # --- PAUSING ---
            self.animation_timer.stop()
            self.play_pause_button.setText("Play Animation")
            logging.info("Animation Paused.")
            # Optionally, release video writer on pause if you only want to record continuous segments
            # if self.video_writer and self.video_writer.isOpened():
            #     logging.info("Releasing video writer on pause.")
            #     self.video_writer.release()
            #     self.video_writer = None 
            #     global _main_app_window_instance_for_atexit # So atexit knows it's handled
            #     _main_app_window_instance_for_atexit = None # If MainAppWindow instance changes or script ends
        else:
            # --- STARTING or RESUMING ---
            if not self.processor.timestamps or len(self.processor.timestamps) == 0:
                logging.warning("No data to animate.")
                self.is_animating = False # Ensure state is correct
                self.play_pause_button.setText("Play Animation")
                return

            # Initialize or re-initialize video writer if not already open
            # This allows recording to start/resume when play is pressed
            if not hasattr(self, 'video_writer') or self.video_writer is None or not self.video_writer.isOpened():
                if not self._initialize_video_writer(): # Try to initialize
                    logging.warning("Video writer could not be initialized. Animation will play without recording.")
                # If _initialize_video_writer fails, self.video_writer will be None,
                # and the compositing step will safely skip writing frames.
            
            # Reset current_timestamp_idx to 0 if you want "Play" to always restart from beginning
            # self.current_timestamp_idx = 0 
            # Or, to resume, just ensure it's valid:
            if self.current_timestamp_idx >= len(self.processor.timestamps):
                self.current_timestamp_idx = 0


            self.animation_timer.start(int(1000 / self.fps))
            self.play_pause_button.setText("Pause Animation")
            logging.info(f"Animation Started/Resumed at {self.fps} FPS.")
        
        self.is_animating = not self.is_animating

    def animation_step(self): 
        if not self.processor.timestamps: 
            self.toggle_animation() # Stop animation if no timestamps
            return
        
        # Ensure current_timestamp_idx is valid
        if self.current_timestamp_idx >= len(self.processor.timestamps):
            self.current_timestamp_idx = 0 # Loop or stop based on your preference

        current_timestamp = self.processor.timestamps[self.current_timestamp_idx]
        self.last_animated_timestamp = current_timestamp # Store for other methods if needed
        
        # Update all visualizers to the current timestamp
        self.vedo_grid_widget.update_view(current_timestamp)
        self.vedo_3d_bar_widget.update_view(current_timestamp)
        
        if self.graph_visualizer.figure and self.graph_visualizer.ax:
            self.graph_visualizer.update_graph_to_timestamp(current_timestamp, self.currently_graphed_tooth_ids)
            self.graph_visualizer.update_time_indicator(current_timestamp) 
            self.graph_qt_canvas.draw_idle() # Request Matplotlib to redraw its canvas for live view
        
        # --- Composite and Write Video Frame ---
        if self.video_writer and self.video_writer.isOpened():
            # 1. Get frames from each visualizer
            frame_grid = self.vedo_grid_widget.get_frame_as_array(current_timestamp)
            frame_3d_bar = self.vedo_3d_bar_widget.get_frame_as_array(current_timestamp)
            frame_graph = self.graph_visualizer.get_frame_as_array(current_timestamp, self.currently_graphed_tooth_ids)

            # 2. Create canvas
            # Ensure canvas is BGR for OpenCV if input frames are BGR
            canvas = np.full((self.canvas_height, self.canvas_width, 3), (210, 210, 210), dtype=np.uint8) # Light grey

            # 3. Define layout regions (x, y, width, height) - TUNE THESE!
            # Example: Vedo views side-by-side on top, Graph at bottom
            h_top_row = int(self.canvas_height * 0.60)  # Height for the row containing Vedo views
            w_vedo_panel = int(self.canvas_width / 2) # Width for each Vedo view panel
            
            h_graph_row = self.canvas_height - h_top_row # Remaining height for the graph
            w_graph_panel = self.canvas_width          # Graph takes full width at the bottom

            # Top-left for Grid View
            if frame_grid is not None:
                if frame_grid.shape[0] > 0 and frame_grid.shape[1] > 0: # Check if frame is valid
                    frame_grid_resized = cv2.resize(frame_grid, (w_vedo_panel, h_top_row))
                    canvas[0:h_top_row, 0:w_vedo_panel] = frame_grid_resized
                else: logging.warning("Grid frame is empty or invalid.")
            
            # Top-right for 3D Bar View
            if frame_3d_bar is not None:
                if frame_3d_bar.shape[0] > 0 and frame_3d_bar.shape[1] > 0:
                    frame_3d_bar_resized = cv2.resize(frame_3d_bar, (w_vedo_panel, h_top_row))
                    canvas[0:h_top_row, w_vedo_panel:w_vedo_panel*2] = frame_3d_bar_resized # Corrected end x-coordinate
                else: logging.warning("3D Bar frame is empty or invalid.")

            # Bottom for Graph View
            if frame_graph is not None:
                if frame_graph.shape[0] > 0 and frame_graph.shape[1] > 0:
                    frame_graph_resized = cv2.resize(frame_graph, (w_graph_panel, h_graph_row))
                    canvas[h_top_row : self.canvas_height, 0:w_graph_panel] = frame_graph_resized
                else: logging.warning("Graph frame is empty or invalid.")
            
            self.video_writer.write(canvas) # Write the composited frame
        # --- End Compositing ---
        
        self.current_timestamp_idx = (self.current_timestamp_idx + 1)
        if self.current_timestamp_idx >= len(self.processor.timestamps):
            self.current_timestamp_idx = 0 # Loop animation
            # If you don't want to loop, stop the timer:
            # self.toggle_animation() 
            # logging.info("Animation sequence finished.")

        logging.debug(f"Qt App Step: Time {current_timestamp:.1f}s")

    def update_graph_on_click(self, sel_tid=None): # ... (same logic)
        new_ids = [sel_tid] if sel_tid is not None else self.initial_graph_teeth
        if new_ids!=self.currently_graphed_tooth_ids or not self.graph_visualizer.lines:
            self.graph_visualizer.plot_tooth_lines(new_ids); self.currently_graphed_tooth_ids=new_ids
            if self.processor.timestamps:
                curr_t = self.processor.timestamps[self.current_timestamp_idx]
                self.graph_visualizer.update_graph_to_timestamp(curr_t,new_ids)
                self.graph_visualizer.update_time_indicator(curr_t)
            self.graph_qt_canvas.draw_idle()
            
    def update_detailed_info(self, info_str): self.detailed_info_label.setText(info_str)

    def reset_3d_bar_camera(self):
        """Calls the reset camera method on the 3D bar visualizer."""
        if self.vedo_3d_bar_widget and hasattr(self.vedo_3d_bar_widget.visualizer, 'reset_camera_view'):
            logging.info("Resetting 3D Bar View camera.")
            self.vedo_3d_bar_widget.visualizer.reset_camera_view()
            # The EmbeddedVedoWidget might need an explicit update call if its own Render isn't enough
            if hasattr(self.vedo_3d_bar_widget, 'Render'):
                self.vedo_3d_bar_widget.Render()
        else:
            logging.warning("Could not reset 3D bar camera - visualizer or method missing.")
            
    def closeEvent(self, event): # ... (same as before) ...
        logging.info("Main window closing..."); self.animation_timer.stop()
        if hasattr(self, 'video_writer') and self.video_writer and self.video_writer.isOpened():
            logging.info("Releasing video writer from MainAppWindow closeEvent.")
            self.video_writer.release(); self.video_writer = None
            global _video_writer_for_atexit; _video_writer_for_atexit = None 
        super().closeEvent(event)

if __name__ == '__main__': # ... (same as before) ...
    app = QApplication(sys.argv)
    reader=SensorDataReader(); data=reader.simulate_data(duration=10,num_teeth=16,num_sensor_points_per_tooth=4)
    processor=DataProcessor(data); processor.create_force_matrix()
    if not processor.timestamps: logging.error("No timestamps. Exiting."); sys.exit(-1)
    main_window = MainAppWindow(processor) 
    main_window.show()
    sys.exit(app.exec_())
# --- END OF FILE main_qt_app.py ---