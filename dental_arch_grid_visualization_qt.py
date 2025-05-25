# --- START OF FILE dental_arch_grid_visualization_qt.py ---
import numpy as np
from vedo import Text2D, Line, Rectangle, Text3D, Grid, Sphere, colors # Plotter is now passed
import logging
import vtk # Add this import at the top of dental_arch_grid_visualization_qt.py


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DentalArchGridVisualizerQt:
    def __init__(self, processor, plotter_instance): 
        self.processor = processor
        self.plotter = plotter_instance 
        if self.plotter and self.plotter.interactor:
            self.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyleImage())
    # This style is strictly for 2D images: allows pan (middle mouse/Shift+LMB) & zoom (RMB drag/wheel)
    # It prevents any rotation.

        if self.processor.cleaned_data is None: self.processor.create_force_matrix()
        self.num_data_teeth = len(self.processor.tooth_ids) if self.processor.tooth_ids else 0
        
        self.arch_layout_width = 16.0; self.arch_layout_depth = 10.0
        self.tooth_cell_definitions = {} 
        self.max_force_for_scaling = self.processor.max_force_overall if hasattr(self.processor,'max_force_overall') else 100.0
        
        if self.plotter:
            self.plotter.camera.ParallelProjectionOn()
            # To further restrict any accidental 3D-like camera movements for a 2D view:
            self.plotter.interactor.viewer = self.plotter # Link interactor to plotter
            self.plotter.interactor.SetInteractorStyle(vtk.vtkInteractorStyleImage()) # Strictly 2D interaction
            # Or for slightly more flexible 2D panning/zooming without 3D rotation:
            # self.plotter.interactor_style = 0 # Default 2D interaction
            # self.plotter.camera.FixedPositionOn() # May help lock camera orientation
            # self.plotter.camera.FixedFocalPointOn()
            self._fit_camera_to_grid()


        self.timestamps = self.processor.timestamps; self.current_timestamp_idx = 0; self.last_animated_timestamp = None
        self.selected_tooth_id_grid = None
        self.grid_outline_actors = {}; self.tooth_label_actors = {} 
        
        # --- Use consistent name ---
        self.intra_tooth_heatmap_actors_list = [] 
        self.force_percentage_actors_list = []    
        self.force_percentage_bg_actors_list = [] 
        # --- End consistent name ---

        self.left_right_bar_actor_left = None; self.left_right_bar_actor_right = None
        self.left_bar_label_actor = None; self.right_bar_label_actor = None
        self.left_bar_percentage_actor = None; self.right_bar_percentage_actor = None
        self.cof_trajectory_line_actor = None; self.cof_current_marker_actor = None; self.time_text_actor = None
        self.selected_tooth_info_text_actor = None        
        self.animation_controller_ref = None

        if self.num_data_teeth > 0:
            self.tooth_cell_definitions = self._define_explicit_tscan_layout(self.num_data_teeth)
            self._initialize_static_grid_elements() 
            if self.tooth_cell_definitions: self.processor.calculate_cof_trajectory(self.tooth_cell_definitions)
            if self.plotter: self.plotter.add_callback('mouse click', self._on_mouse_click)
        else:
            logging.error("GridVizQt: No tooth data to initialize.")
            if self.plotter: self.plotter.add(Text2D("No Grid Data Available", c='red', s=1.5))

    def _initialize_empty_state(self):
        self.tooth_cell_definitions={}; self.timestamps=[]
        self.current_timestamp_idx=0; self.last_animated_timestamp=None; 
        # --- Use consistent name ---
        self.intra_tooth_heatmap_actors_list=[] 
        self.tooth_label_actors={}; 
        self.force_percentage_actors_list=[]; 
        self.time_text_actor=None; 
        self.grid_outline_actors={}; 
        self.force_percentage_bg_actors_list=[] 
        # --- End consistent name ---
        self.left_right_bar_actor_left=None;self.left_right_bar_actor_right=None;self.left_bar_label_actor=None
        self.right_bar_label_actor=None;self.left_bar_percentage_actor=None;self.right_bar_percentage_actor=None
        self.cof_trajectory_line_actor=None;self.cof_current_marker_actor=None;self.selected_tooth_info_text_actor=None
        self.selected_tooth_id_grid=None;self.animation_controller_ref=None

    def set_animation_controller_for_graph_link(self, controller):
        """Stores a reference to the MainAppWindow (which acts as a controller)."""
        self.animation_controller_ref = controller

    def _initialize_static_grid_elements(self):
        if not self.tooth_cell_definitions or not self.plotter: return
        
        # Clear any previous static actors if this method were ever called again
        if self.grid_outline_actors: self.plotter.remove(list(self.grid_outline_actors.values()))
        if self.tooth_label_actors: self.plotter.remove(list(self.tooth_label_actors.values()))
        self.grid_outline_actors.clear()
        self.tooth_label_actors.clear()

        actors_to_add_at_once = []
        for _layout_idx, cell_prop in self.tooth_cell_definitions.items(): # Iterates if dict
            tooth_id = cell_prop['actual_id']
            cx, cy = cell_prop['center']; w, h = cell_prop['width'], cell_prop['height']
            p1, p2 = (cx - w/2, cy - h/2), (cx + w/2, cy + h/2)
            
            outline = Rectangle(p1, p2, c=(0.3,0.3,0.3), alpha=0.8); outline.lw(1.0)
            outline.name = f"Outline_Tooth_{tooth_id}"; outline.pickable = True
            self.grid_outline_actors[tooth_id] = outline 
            actors_to_add_at_once.append(outline)

            label_pos_xy = (cx, cy + h * 0.5 + 0.2)
            text_s = h * 0.30; text_s = max(0.25, min(text_s, 0.5)) 
            label = Text3D(str(tooth_id), pos=(label_pos_xy[0], label_pos_xy[1], 0.12), 
                           s=text_s, c=(0.05,0.05,0.3), justify='center-center', depth=0.01) 
            self.tooth_label_actors[tooth_id] = label 
            actors_to_add_at_once.append(label)
        
        if actors_to_add_at_once:
            self.plotter.add(actors_to_add_at_once)

    def _fit_camera_to_grid(self):
        if not self.tooth_cell_definitions or not self.plotter: return
        all_props = list(self.tooth_cell_definitions.values())
        if not all_props: self.plotter.camera.SetParallelScale(10); return
        all_x = [p['center'][0]+p['width']/2 for p in all_props]+[p['center'][0]-p['width']/2 for p in all_props]
        all_y = [p['center'][1]+p['height']/2 for p in all_props]+[p['center'][1]-p['height']/2 for p in all_props]
        min_y_bottom = min(p['center'][1]-p['height']/2 for p in all_props) if all_props else 0
        all_y.append(min_y_bottom - 3.0) 
        if not all_x or not all_y: self.plotter.camera.SetParallelScale(10); return
        min_x,max_x=min(all_x),max(all_x); min_y,max_y=min(all_y),max(all_y)
        pad=1.15; vh=(max_y-min_y)*pad; vw=(max_x-min_x)*pad
        scale=max(vh,vw,1.0)/2.0; scale=max(0.1,scale)
        self.plotter.camera.SetParallelScale(scale)
        fx,fy=(min_x+max_x)/2,(min_y+max_y)/2
        self.plotter.camera.SetFocalPoint(fx,fy,0); self.plotter.camera.SetPosition(fx,fy,10)

    def _get_arch_positions_for_layout(self, num_teeth, arch_width, arch_depth):
        if num_teeth == 0: return np.array([])
        x_coords = np.array([0.0]) if num_teeth==1 else np.linspace(-arch_width/2,arch_width/2,num_teeth)
        k = arch_depth/((arch_width/2)**2) if arch_width!=0 else 0
        return np.array([[x, arch_depth - k*(x**2), 0] for x in x_coords])

    def _define_explicit_tscan_layout(self, num_teeth_from_data):
        layout = {} 
        if num_teeth_from_data == 0 or not self.processor.tooth_ids: return layout
        # ... (Your carefully tuned layout logic from before - this needs to be accurate) ...
        # Using the programmatic U-shape as a base:
        base_arch_w_centers=self.arch_layout_width*0.80; base_arch_d_centers=self.arch_layout_depth*0.70
        arch_centers_3d=self._get_arch_positions_for_layout(num_teeth_from_data,base_arch_w_centers,base_arch_d_centers)
        arch_centers_xy=[ac[:2] for ac in arch_centers_3d]
        if num_teeth_from_data > 1:
            sorted_x=sorted([c[0] for c in arch_centers_xy]); dx=np.abs(np.diff(sorted_x))
            avg_spacing_x=np.mean(dx) if len(dx)>0 else base_arch_w_centers/num_teeth_from_data
            base_w=avg_spacing_x*0.90; base_h=base_w*1.1 
        else: base_w=self.arch_layout_width*0.15; base_h=self.arch_layout_depth*0.15
        base_w=max(0.7,base_w); base_h=max(0.9,base_h)
        for i in range(num_teeth_from_data):
            actual_id=self.processor.tooth_ids[i]; center_xy=arch_centers_xy[i]
            curr_w=base_w; curr_h=base_h
            norm_x=abs(center_xy[0])/(base_arch_w_centers/2.0) if base_arch_w_centers>0 else 0
            w_scale=1.0; h_scale=1.0
            if norm_x>0.75:w_scale=1.35;h_scale=0.85 
            elif norm_x>0.50:w_scale=1.1;h_scale=1.0
            elif norm_x<0.10:w_scale=0.70;h_scale=1.20 
            elif norm_x<0.35:w_scale=0.85;h_scale=1.10
            final_w=curr_w*w_scale; final_h=curr_h*h_scale
            final_w=max(0.6,final_w); final_h=max(0.8,final_h)
            layout[i]={'center':center_xy,'width':final_w,'height':final_h,'actual_id':actual_id}
        return layout

    def _create_intra_tooth_heatmap(self, cell_prop, forces_on_this_tooth_sensors):
        if not forces_on_this_tooth_sensors: 
            return None
        
        # --- DEFINE num_sp AT THE BEGINNING ---
        num_sp = len(forces_on_this_tooth_sensors)
        if num_sp == 0: 
            return None
        # --- END DEFINITION ---
        
        cx, cy = cell_prop['center']
        cw, ch = cell_prop['width'], cell_prop['height']
        
        heatmap_grid_resolution_param = (1, 1) 
        heatmap_grid = Grid(s=[cw * 0.96, ch * 0.96], res=heatmap_grid_resolution_param)
        heatmap_grid.pos(cx, cy, 0.05).lw(0).alpha(0.95) # z=0.05 to be above outline slightly
        heatmap_grid.name = f"Heatmap_Tooth_{cell_prop['actual_id']}"
        heatmap_grid.pickable = True # Make heatmap itself pickable
        
        grid_points_forces = np.zeros(heatmap_grid.npoints) # Should be 4 points for res=(1,1)

        # Mapping 4 sensor forces to the 4 points of the Grid (res=(1,1))
        # Vedo Grid point order for res=(1,1) (a single cell with 4 points):
        # Point 0: Bottom-Left (-sx/2, -sy/2) relative to grid center
        # Point 1: Bottom-Right ( sx/2, -sy/2)
        # Point 2: Top-Left    (-sx/2,  sy/2)
        # Point 3: Top-Right   ( sx/2,  sy/2)
        # Assuming sensor_point_id 1, 2, 3, 4 in data map to a visual TL, TR, BL, BR
        if num_sp == 4: # Ensure we have exactly 4 forces for this mapping
            grid_points_forces[2] = forces_on_this_tooth_sensors.get(1, 0.0) # Sensor 1 (TL) -> Grid Point 2 (Top-Left)
            grid_points_forces[3] = forces_on_this_tooth_sensors.get(2, 0.0) # Sensor 2 (TR) -> Grid Point 3 (Top-Right)
            grid_points_forces[0] = forces_on_this_tooth_sensors.get(3, 0.0) # Sensor 3 (BL) -> Grid Point 0 (Bottom-Left)
            grid_points_forces[1] = forces_on_this_tooth_sensors.get(4, 0.0) # Sensor 4 (BR) -> Grid Point 1 (Bottom-Right)
        elif num_sp > 0: # Fallback if not exactly 4 sensor points (e.g. 1, 2, or 3)
            avg_force = np.mean(list(forces_on_this_tooth_sensors.values()))
            grid_points_forces.fill(avg_force)
        # If num_sp is 0, grid_points_forces remains all zeros, resulting in 'darkblue'

        heatmap_grid.pointdata["forces"] = np.nan_to_num(grid_points_forces)
        custom_cmap_rgb = ['darkblue', (0,0,1), (0,1,0), (1,1,0), (1,0,0)] 
        heatmap_grid.cmap(custom_cmap_rgb, "forces", vmin=0, vmax=self.max_force_for_scaling)
        
        return heatmap_grid

    def render_arch(self, timestamp):
        if not self.tooth_cell_definitions or not self.plotter: 
            if self.plotter and self.plotter.window: self.plotter.render()
            return
        
        # 1. Clear all previously added dynamic actors for this frame
        actors_to_remove = []
        if self.time_text_actor: actors_to_remove.append(self.time_text_actor)
        actors_to_remove.extend(self.intra_tooth_heatmap_actors_list)
        actors_to_remove.extend(self.force_percentage_bg_actors_list)
        actors_to_remove.extend(self.force_percentage_actors_list)
        if self.left_right_bar_actor_left: actors_to_remove.append(self.left_right_bar_actor_left)
        if self.left_right_bar_actor_right: actors_to_remove.append(self.left_right_bar_actor_right)
        if self.left_bar_label_actor: actors_to_remove.append(self.left_bar_label_actor)
        if self.right_bar_label_actor: actors_to_remove.append(self.right_bar_label_actor)
        if self.left_bar_percentage_actor: actors_to_remove.append(self.left_bar_percentage_actor)
        if self.right_bar_percentage_actor: actors_to_remove.append(self.right_bar_percentage_actor)
        if self.cof_trajectory_line_actor: actors_to_remove.append(self.cof_trajectory_line_actor)
        if self.cof_current_marker_actor: actors_to_remove.append(self.cof_current_marker_actor)
        if self.selected_tooth_info_text_actor: actors_to_remove.append(self.selected_tooth_info_text_actor)
        
        actors_to_remove_filtered = [a for a in actors_to_remove if a is not None]
        if actors_to_remove_filtered: self.plotter.remove(actors_to_remove_filtered)

        # Reset instance lists/actor references for actors recreated each frame
        self.intra_tooth_heatmap_actors_list.clear() 
        self.force_percentage_bg_actors_list.clear() 
        self.force_percentage_actors_list.clear()
        self.time_text_actor=None; self.left_right_bar_actor_left=None; self.left_right_bar_actor_right=None
        self.left_bar_label_actor=None; self.right_bar_label_actor=None; self.left_bar_percentage_actor=None; self.right_bar_percentage_actor=None
        self.cof_trajectory_line_actor=None; self.cof_current_marker_actor=None; self.selected_tooth_info_text_actor = None
        
        # This list will collect all new actors to be added in THIS specific call to render_arch
        current_actors_to_add = [] 
        
        self.time_text_actor = Text2D(f"Time: {timestamp:.1f}s",pos="bottom-left",c='k',bg=(1,1,1),alpha=0.7,s=0.7)
        current_actors_to_add.append(self.time_text_actor)
        
        ordered_pairs, forces_all_sensor_points = self.processor.get_all_forces_at_time(timestamp)
        if not ordered_pairs: 
            if current_actors_to_add and self.plotter: self.plotter.add(current_actors_to_add)
            return
        
        force_map_all_sensors = {p:f for p,f in zip(ordered_pairs,forces_all_sensor_points)}
        
        # --- INITIALIZE force_left_side, force_right_side, and total_force_on_arch_this_step HERE ---
        total_force_on_arch_this_step = sum(f for f in forces_all_sensor_points if np.isfinite(f) and f > 0)
        total_force_on_arch_this_step = max(total_force_on_arch_this_step, 1e-6) # Avoid division by zero
        force_left_side = 0.0
        force_right_side = 0.0
        # --- END INITIALIZATION ---

        for _layout_idx, cell_prop in self.tooth_cell_definitions.items():
            tooth_id = cell_prop['actual_id']
            
            outline_actor = self.grid_outline_actors.get(tooth_id) 
            if outline_actor:
                if self.selected_tooth_id_grid == tooth_id: outline_actor.color('lime').lw(3.0).alpha(1.0) 
                else: outline_actor.color((0.3,0.3,0.3)).lw(1.0).alpha(0.8)
            
            current_tooth_total_force = 0.0
            sensor_ids_for_this_tooth = [spid for tid,spid in self.processor.ordered_tooth_sensor_pairs if tid==tooth_id]
            forces_on_this_tooth_sensors = {}
            for sp_id in sensor_ids_for_this_tooth:
                force = force_map_all_sensors.get((tooth_id,sp_id),0.0); force=np.nan_to_num(force)
                forces_on_this_tooth_sensors[sp_id]=force; current_tooth_total_force+=force
            
            # Accumulate Left/Right forces
            if cell_prop['center'][0] < -0.01: force_right_side += current_tooth_total_force 
            elif cell_prop['center'][0] > 0.01: force_left_side += current_tooth_total_force  
            else: force_left_side+=current_tooth_total_force/2.0; force_right_side+=current_tooth_total_force/2.0
            
            heatmap_actor = self._create_intra_tooth_heatmap(cell_prop, forces_on_this_tooth_sensors)
            if heatmap_actor:
                if self.selected_tooth_id_grid == tooth_id: heatmap_actor.alpha(1.0)
                else: heatmap_actor.alpha(0.75)
                self.intra_tooth_heatmap_actors_list.append(heatmap_actor) 
            
            perc = (current_tooth_total_force/total_force_on_arch_this_step)*100
            text_s = cell_prop['height']*0.20; text_s = max(0.20,min(text_s,0.45)) 
            perc_pos_xy = (cell_prop['center'][0],cell_prop['center'][1]-cell_prop['height']*0.70); pz = 0.16 
            num_chars=len(f"{perc:.1f}%"); bg_w_est=text_s*num_chars*0.50; bg_h_est=text_s*1.0
            bg_w_est=max(cell_prop['width']*0.25,bg_w_est); bg_h_est=max(cell_prop['height']*0.15,bg_h_est)
            p1_bg=(perc_pos_xy[0]-bg_w_est/2,perc_pos_xy[1]-bg_h_est/2);p2_bg=(perc_pos_xy[0]+bg_w_est/2,perc_pos_xy[1]+bg_h_est/2)
            pbg_rgb=(0.95,0.95,0.85);pbg_a=0.75
            p_bg=Rectangle(p1_bg,p2_bg,c=pbg_rgb,alpha=pbg_a);p_bg.z(pz-0.02) 
            self.force_percentage_bg_actors_list.append(p_bg)
            p_lbl=Text3D(f"{perc:.1f}%",pos=(perc_pos_xy[0],perc_pos_xy[1],pz),s=text_s,c='k',justify='cc',depth=0.01) 
            self.force_percentage_actors_list.append(p_lbl)
        
        current_actors_to_add.extend(self.intra_tooth_heatmap_actors_list)
        current_actors_to_add.extend(self.force_percentage_bg_actors_list)
        current_actors_to_add.extend(self.force_percentage_actors_list)

        # Now use the fully accumulated force_left_side and force_right_side
        perc_l=(force_left_side/total_force_on_arch_this_step)*100
        perc_r=(force_right_side/total_force_on_arch_this_step)*100
        
        if self.tooth_cell_definitions: # This check might be redundant if the outer one passed
            min_y_overall=min(p['center'][1]-p['height']/2 for p in self.tooth_cell_definitions.values())
            bar_base_y=min_y_overall-1.8; bar_overall_width=self.arch_layout_width*0.30; bar_max_h=0.8 
            left_bar_h=max(0.02,(perc_l/100.0)*bar_max_h); right_bar_h=max(0.02,(perc_r/100.0)*bar_max_h)
            bar_label_s=0.25; bar_perc_s=0.22 # Adjusted sizes from previous fix
            bar_z=0.05; text_on_bar_z=bar_z+0.02; label_above_bar_z=bar_z+0.03

            l_bar_cx=-bar_overall_width*0.8; 
            l_p1=(l_bar_cx-bar_overall_width/2,bar_base_y); l_p2=(l_bar_cx+bar_overall_width/2,bar_base_y+left_bar_h)
            self.left_right_bar_actor_left=Rectangle(l_p1,l_p2,c='g',alpha=0.85).z(bar_z)
            left_label_pos=(l_bar_cx,bar_base_y+left_bar_h+0.20,label_above_bar_z); 
            self.left_bar_label_actor=Text3D("Left",pos=left_label_pos,s=bar_label_s,c='k',justify='cb',depth=0.01)
            if left_bar_h > 0.02: 
                left_perc_pos=(l_bar_cx,bar_base_y+left_bar_h/2,text_on_bar_z); 
                self.left_bar_percentage_actor=Text3D(f"{perc_l:.0f}%",pos=left_perc_pos,s=bar_perc_s,c='w',justify='cc',depth=0.01)
            else: self.left_bar_percentage_actor = None
            
            r_bar_cx=bar_overall_width*0.8; 
            r_p1=(r_bar_cx-bar_overall_width/2,bar_base_y); r_p2=(r_bar_cx+bar_overall_width/2,bar_base_y+right_bar_h)
            self.left_right_bar_actor_right=Rectangle(r_p1,r_p2,c='r',alpha=0.85).z(bar_z)
            right_label_pos=(r_bar_cx,bar_base_y+right_bar_h+0.20,label_above_bar_z); 
            self.right_bar_label_actor=Text3D("Right",pos=right_label_pos,s=bar_label_s,c='k',justify='cb',depth=0.01)
            if right_bar_h > 0.02:
                right_perc_pos=(r_bar_cx,bar_base_y+right_bar_h/2,text_on_bar_z); 
                self.right_bar_percentage_actor=Text3D(f"{perc_r:.0f}%",pos=right_perc_pos,s=bar_perc_s,c='w',justify='cc',depth=0.01)
            else: self.right_bar_percentage_actor = None
            
            current_actors_to_add.extend(filter(None,[
                self.left_right_bar_actor_left,self.left_bar_label_actor,self.left_bar_percentage_actor, 
                self.left_right_bar_actor_right,self.right_bar_label_actor,self.right_bar_percentage_actor
            ]))
        
        cof_pts=self.processor.get_cof_up_to_timestamp(timestamp)
        if len(cof_pts)>1: 
            cof_ln_pts=[(p[0],p[1],0.25) for p in cof_pts]
            self.cof_trajectory_line_actor=Line(cof_ln_pts,c=(0.8,0.1,0.8),lw=2,alpha=0.6)
            current_actors_to_add.append(self.cof_trajectory_line_actor)
        if cof_pts: 
            cx_cof,cy_cof=cof_pts[-1]
            self.cof_current_marker_actor=Sphere(pos=(cx_cof,cy_cof,0.27),r=0.10,c='darkred',alpha=0.9)
            current_actors_to_add.append(self.cof_current_marker_actor)
        
        if current_actors_to_add and self.plotter: 
            self.plotter.add(current_actors_to_add)

        # # --- ENSURE THIS IS PRESENT AND UNCONDITIONAL (if plotter exists) ---
        # if self.plotter and self.plotter.window: # Or just self.plotter if offscreen is possible for screenshots
        #     self.plotter.render()


    def animate(self, event=None):
        if not self.timestamps: return
        if self.current_timestamp_idx >= len(self.timestamps): self.current_timestamp_idx = 0
        t = self.timestamps[self.current_timestamp_idx]
        self.last_animated_timestamp = t
        self.render_arch(t) 
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
        self.render_arch(timestamp_to_render) 
        
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

    def _on_mouse_click(self, event):
        # ... (The complete _on_mouse_click from the previous fully working version) ...
        clicked_tooth_id_parsed = None; is_background_click = True
        original_selected_tooth_id = self.selected_tooth_id_grid 
        if event.actor:
            is_background_click = False; actor_name = event.actor.name
            logging.info(f"Grid Plotter Clicked: Actor '{actor_name}' at {event.picked3d}")
            if actor_name and (actor_name.startswith("Heatmap_Tooth_") or actor_name.startswith("Outline_Tooth_")):
                try: clicked_tooth_id_parsed = int(actor_name.split("_")[-1])
                except ValueError: logging.warning(f"Could not parse: {actor_name}")
        else: logging.info("Grid Plotter Clicked: Background")
        if clicked_tooth_id_parsed is not None:
            if self.selected_tooth_id_grid == clicked_tooth_id_parsed: self.selected_tooth_id_grid = None 
            else: self.selected_tooth_id_grid = clicked_tooth_id_parsed
            print(f"--- Tooth {self.selected_tooth_id_grid if self.selected_tooth_id_grid else 'deselected'} ---")
        elif is_background_click: self.selected_tooth_id_grid = None 
        else: self.selected_tooth_id_grid = None
        if self.animation_controller_ref:
            self.animation_controller_ref.update_graph_on_click(self.selected_tooth_id_grid) 
            if self.selected_tooth_id_grid is not None:
                timestamp_for_info = self.last_animated_timestamp 
                if timestamp_for_info is None and self.timestamps and len(self.timestamps) > 0 : timestamp_for_info = self.timestamps[self.current_timestamp_idx] 
                elif timestamp_for_info is None: timestamp_for_info = 0.0
                info_text_lines = [f"Tooth ID: {self.selected_tooth_id_grid}", f"Forces @ {timestamp_for_info:.1f}s:"]
                total_force_on_selected_tooth = 0.0
                sensor_ids = [spid for tid,spid in self.processor.ordered_tooth_sensor_pairs if tid==self.selected_tooth_id_grid]
                _p, forces_ts = self.processor.get_all_forces_at_time(timestamp_for_info); map_ts={p:f for p,f in zip(_p,forces_ts)}
                for sp_id in sensor_ids: force = map_ts.get((self.selected_tooth_id_grid,sp_id),0.0); info_text_lines.append(f" S{sp_id}:{force:.1f}N"); total_force_on_selected_tooth+=force
                info_text_lines.append(f"Total: {total_force_on_selected_tooth:.1f}N"); info_text_full = "\n".join(info_text_lines)
                self.animation_controller_ref.update_detailed_info(info_text_full)
            else: self.animation_controller_ref.update_detailed_info("Click a tooth cell for details.")
        is_timer_active = False
        if self.animation_controller_ref and hasattr(self.animation_controller_ref, 'is_animating'): is_timer_active = self.animation_controller_ref.is_animating
        if not is_timer_active and self.timestamps:
            current_t = self.last_animated_timestamp if self.last_animated_timestamp is not None else (self.timestamps[0] if self.timestamps else 0.0)
            if current_t is not None: self.render_arch(current_t)

    # NO start_animation method, as it's driven by MainAppWindow's QTimer
# --- END OF FILE dental_arch_grid_visualization_qt.py ---