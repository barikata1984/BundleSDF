
import numpy as np
import trimesh
import sys
import os
import time
import threading

# Add path to BundleSDF
sys.path.append("/root/osx-ur/underlay_ws/src/third_party/BundleSDF")
from gui import BundleSdfGui
import dearpygui.dearpygui as dpg

def run_gui_sim():
    print("Starting GUI simulation...")
    gui = BundleSdfGui(img_height=300)
    
    # Create a dummy mesh
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    
    # Initialize dummy K and W
    gui.K = np.array([[100, 0, 150], [0, 100, 150], [0, 0, 1]])
    gui.W = 400

    # Simulate first update
    print("First mesh update...")
    gui.update_mesh(mesh)
    
    # Simulate rendering
    print("Simulating render loop...")
    for i in range(10):
        if gui.renderer:
             gui.update_render_mesh()
        dpg.render_dearpygui_frame()
        time.sleep(0.1)
        
    # Simulate second update (mimicking the crash point)
    print("Second mesh update (larger mesh)...")
    mesh2 = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    # Subdivide to make it more complex like a real NeRF mesh
    for _ in range(2):
        mesh2 = mesh2.subdivide()
        
    gui.update_mesh(mesh2)
    print("Second mesh update done.")
    
    # Render loop again
    print("Render loop after second update...")
    for i in range(50):
        if gui.renderer:
             gui.update_render_mesh()
        dpg.render_dearpygui_frame()
        time.sleep(0.05)

    dpg.destroy_context()
    print("GUI simulation finished successfully.")

if __name__ == "__main__":
    try:
        run_gui_sim()
    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback
        traceback.print_exc()
