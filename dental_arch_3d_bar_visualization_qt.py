# --- START OF FILE dental_arch_3d_bar_visualization_qt.py ---
import numpy as np
from vedo import Plotter, Text2D, Cylinder, Box, Line, Axes, Grid, Plane, Text3D
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DentalArch3DBarVisualizerQt:
    def __init__(self, processor, plotter_instance): # Receives a Vedo Plotter
        self.processor = processor
        self.plotter = plotter_instance # Use the passed plotter

        if self.processor.cleaned_data is None: self.processor.create_force_matrix()
        self.num_data_teeth = len(self.processor.tooth_ids) if self.processor.tooth_ids else 0
        if self.num_data_teeth == 0: logging.error("3DBarVizQt: No tooth data."); self._initialize_empty_state(); return

        self.arch_layout_width = 14.0; self.arch_layout_depth = 8.0; self.bar_base_radius = 0.5     
        self.tooth_bar_base_positions = self._create_bar_base_positions(self.num_data_teeth, self.arch_layout_width, self.arch_layout_depth)
        self.max_force_for_scaling = self.processor.max_force_overall if hasattr(self.processor,'max_force_overall') else 100.0
        self.max_bar_height = 5.0; self.min_bar_height = 0.1 

        # --- Store Initial Camera Parameters ---
        self.initial_camera_settings = {}
        # ---

        # Camera and static elements setup (on self.plotter)
        if self.plotter:
            istyle = self.plotter.interactor.GetInteractorStyle()
            if hasattr(istyle, 'SetMotionFactor'): # Common in trackball styles
                # iStyle.SetMotionFactor(0.0) # Disables motion, might be too restrictive
                pass 
            self.plotter.interactor_style = 1 # JoystickActor: a bit different, might work
            floor_size_x=self.arch_layout_width*1.5; floor_size_y=self.arch_layout_depth*1.8
            floor_grid_res=(10,10); floor_grid=Grid(s=(floor_size_x,floor_size_y),res=floor_grid_res,c='gainsboro',alpha=0.4)
            self.grid_center_y=-self.arch_layout_depth*0.3; floor_grid.pos(0,self.grid_center_y,-0.05)
            self.plotter.add(floor_grid)
             # Initialize and store camera settings
            self.reset_camera_view() # Call to set and store initial view

            if self.plotter.actors: self.axes_actor=Axes(self.plotter.actors,c='dimgrey',yzgrid=False); self.plotter.add(self.axes_actor)
            else: self.plotter.add_global_axes(axtype=1,c='dimgrey')
            self.plotter.camera.SetPosition(0,-self.arch_layout_depth*2.5,self.max_bar_height*2.2) 
            self.plotter.camera.SetFocalPoint(0,self.grid_center_y,self.max_bar_height/3); self.plotter.camera.SetViewUp(0,0.3,0.7) 

        self.timestamps = self.processor.timestamps; self.current_timestamp_idx = 0; self.last_animated_timestamp = None 
        self.force_bar_actors = []; self.time_text_actor = None; self.arch_base_line_actor = None; self.tooth_label_actors = []
        
        if self.num_data_teeth > 0: self._initialize_static_elements()

    def _initialize_empty_state(self): # Not strictly needed if __init__ handles no data
        self.tooth_bar_base_positions=[]; self.timestamps=[]
        self.current_timestamp_idx=0; self.last_animated_timestamp=None; self.force_bar_actors=[]
        self.time_text_actor=None; self.arch_base_line_actor=None; self.tooth_label_actors=[]

    def _create_bar_base_positions(self, num_teeth, total_width, total_depth): # Same
        if num_teeth == 0: return []
        x_coords = np.array([0.0]) if num_teeth==1 else np.linspace(-total_width/2,total_width/2,num_teeth)
        k = total_depth/((total_width/2)**2) if total_width!=0 else 0
        return [np.array([x,total_depth-k*(x**2)-total_depth*0.8,0.0]) for x in x_coords]

    def _initialize_static_elements(self): # Draws on self.plotter
        if not self.plotter: return
        static_actors_to_add = []
        if len(self.tooth_bar_base_positions)>1:
            line_pts=[(p[0],p[1],0.01) for p in self.tooth_bar_base_positions]
            self.arch_base_line_actor=Line(line_pts,c='dimgray',lw=3,alpha=0.8); static_actors_to_add.append(self.arch_base_line_actor)
            self.tooth_label_actors=[]
            for i,pos in enumerate(self.tooth_bar_base_positions):
                if i < len(self.processor.tooth_ids):
                    tid=self.processor.tooth_ids[i]
                    lbl_pos=(pos[0],pos[1]+self.bar_base_radius*0.5,-0.2) 
                    lbl=Text3D(str(tid),pos=lbl_pos,s=0.22,c=(0.1,0.1,0.1),depth=0.01,justify='ct')
                    self.tooth_label_actors.append(lbl)
            if self.tooth_label_actors: static_actors_to_add.extend(self.tooth_label_actors)
        if static_actors_to_add: self.plotter.add(static_actors_to_add)

    def render_display(self, timestamp): # Updates actors on self.plotter
        if not self.tooth_bar_base_positions or not self.plotter: return
        actors_to_remove = []
        if self.time_text_actor: actors_to_remove.append(self.time_text_actor)
        if self.force_bar_actors: actors_to_remove.extend(self.force_bar_actors)
        if actors_to_remove: self.plotter.remove(actors_to_remove)
        self.force_bar_actors.clear(); self.time_text_actor = None
        
        current_actors_to_add = []
        self.time_text_actor = Text2D(f"Time: {timestamp:.1f}s", pos="bottom-right", c='k', bg=(1,1,1), alpha=0.6, s=0.8)        
        current_actors_to_add.append(self.time_text_actor)

        for i,base_pos in enumerate(self.tooth_bar_base_positions):
            if i >= len(self.processor.tooth_ids): continue
            tid=self.processor.tooth_ids[i]; _,f_series=self.processor.get_average_force_for_tooth(tid)
            curr_f=0.0
            if self.timestamps and len(f_series)==len(self.timestamps):
                try: idx=np.argmin(np.abs(np.array(self.timestamps)-timestamp)); curr_f=f_series[idx]
                except Exception: pass 
            if not np.isfinite(curr_f): curr_f=0.0
            norm_f=min(1.0,max(0.0,curr_f/self.max_force_for_scaling))
            bar_h=self.min_bar_height+norm_f*(self.max_bar_height-self.min_bar_height)
            if curr_f<1e-3: bar_h=0.0 
            if bar_h>1e-4: 
                if norm_f<0.01:clr=(0.1,0.1,0.6)    
                elif norm_f<0.25:clr=(0.2,0.4,1)  
                elif norm_f<0.5:clr=(0.1,0.8,0.4) 
                elif norm_f<0.75:clr=(1,0.9,0.1)    
                elif norm_f<0.9:clr=(1,0.4,0)  
                else: clr=(0.9,0.0,0.2)                    
                bar_cz=base_pos[2]+bar_h/2.0
                bar=Box(pos=(base_pos[0],base_pos[1],bar_cz),length=self.bar_base_radius*1.7,width=self.bar_base_radius*1.7,height=bar_h,c=clr,alpha=0.92)
                bar.name=f"Bar_Tooth_{tid}"; bar.pickable=True
                self.force_bar_actors.append(bar); current_actors_to_add.append(bar)
        if current_actors_to_add: self.plotter.add(current_actors_to_add)
        # if self.plotter and self.plotter.window: # Or just self.plotter if offscreen is possible for screenshots
        #     self.plotter.render()
        # self.plotter.render() # QtPlotter handles rendering

    def animate(self, event=None): # Called by QTimer via MainAppWindow
        if not self.timestamps: return
        if self.current_timestamp_idx >= len(self.timestamps): self.current_timestamp_idx = 0
        
        t = self.timestamps[self.current_timestamp_idx]
        self.last_animated_timestamp = t
        self.render_display(t) 
        
        self.current_timestamp_idx = (self.current_timestamp_idx + 1) % len(self.timestamps)

    def get_frame_as_array(self, timestamp_to_render):
        if not self.plotter: 
            logging.warning(f"GRIDVIZ: Plotter not available for screenshot at T={timestamp_to_render}")
            return None

        # 1. Ensure the visualizer's actors are updated for this specific timestamp.
        #    Temporarily set the state for this render, then restore.
        original_anim_idx = self.current_timestamp_idx
        original_last_ts = self.last_animated_timestamp

        render_idx = original_anim_idx 
        if self.timestamps and len(self.timestamps) > 0:
            try:
                render_idx = np.argmin(np.abs(np.array(self.timestamps) - timestamp_to_render))
            except Exception as e:
                logging.debug(f"GRIDVIZ: Error finding index for timestamp {timestamp_to_render}: {e}")
        
        self.current_timestamp_idx = render_idx
        self.last_animated_timestamp = timestamp_to_render
        
        # This call updates the actors on self.plotter but SHOULD NOT call self.plotter.render()
        # The render call will be done explicitly below.
        self.render_display(timestamp_to_render) 
        
        # 2. Explicitly make this plotter's render window current and render IT.
        img_array = None
        if self.plotter.renderer and self.plotter.window: 
            # Get the specific VTK RenderWindow for this plotter
            render_window = self.plotter.renderer.GetRenderWindow()
            
            # --- Critical Section for Screenshotting a Specific Plotter ---
            # This sequence attempts to ensure this plotter's context is active
            # for the screenshot operation.
            # If self.plotter.offscreen was True, these might not be needed or behave differently.
            # We assume self.plotter.offscreen is False for the live view.
            
            # render_window.MakeCurrent() # We found this caused issues with closing other windows. AVOID.

            # Instead, ensure its content is flushed to its buffer by rendering it.
            logging.debug(f"GRIDVIZ: Explicitly rendering plotter '{self.plotter.title}' for screenshot.")
            self.plotter.render() # Render this specific plotter's scene.

            # Now take the screenshot from this plotter
            img_array = self.plotter.screenshot(asarray=True)
            # --- End Critical Section ---
        else:
            logging.warning(f"GRIDVIZ: Plotter for '{self.plotter.title}' has no window or renderer for screenshot. Screenshot might be blank.")

        # 3. Restore original animation state for the live interactive view
        self.current_timestamp_idx = original_anim_idx
        self.last_animated_timestamp = original_last_ts
        
        if img_array is None:
            logging.warning(f"GRIDVIZ: Screenshot returned None for T={timestamp_to_render:.1f}s from plotter '{self.plotter.title}'")
        return img_array

    def reset_camera_view(self):
        """Sets the camera to a predefined initial view and stores these settings."""
        if not self.plotter: return

        # Define your desired initial camera parameters here
        pos = (0, -self.arch_layout_depth * 2.5, self.max_bar_height * 2.2)
        focal_point = (0, self.grid_center_y if hasattr(self, 'grid_center_y') else 0, self.max_bar_height / 3)
        view_up = (0, 0.3, 0.7) # Or (0,1,0) or (0,0,1) depending on desired orientation

        self.plotter.camera.SetPosition(pos)
        self.plotter.camera.SetFocalPoint(focal_point)
        self.plotter.camera.SetViewUp(view_up)
        self.plotter.reset_camera() # Important: applies settings and fits content

        # Store these settings
        self.initial_camera_settings = {
            'position': pos,
            'focal_point': focal_point,
            'viewup': view_up,
            # 'distance': self.plotter.camera.GetDistance(), # Optional
            # 'clipping_range': self.plotter.camera.GetClippingRange(), # Optional
        }
        logging.info(f"3DBarView: Camera reset to initial settings: {self.initial_camera_settings}")
        if self.plotter.window and self.plotter.renderer: # Ensure render if window exists
            self.plotter.render()

    def apply_camera_settings(self, settings):
        """Applies a given set of camera settings."""
        if not self.plotter or not settings: return
        self.plotter.camera.SetPosition(settings.get('position', self.plotter.camera.GetPosition()))
        self.plotter.camera.SetFocalPoint(settings.get('focal_point', self.plotter.camera.GetFocalPoint()))
        self.plotter.camera.SetViewUp(settings.get('viewup', self.plotter.camera.GetViewUp()))
        # if 'distance' in settings: self.plotter.camera.SetDistance(settings['distance'])
        # if 'clipping_range' in settings: self.plotter.camera.SetClippingRange(settings['clipping_range'])
        self.plotter.reset_camera() # Re-apply and fit
        if self.plotter.window and self.plotter.renderer:
            self.plotter.render()

# --- END OF FILE dental_arch_3d_bar_visualization_qt.py ---