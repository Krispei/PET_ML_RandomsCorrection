# main.py
import opengate as gate
import argparse
from build_scanner import build_pet_geometry
from test_phantom import build_test_phantom
import simulation_setup as setup  # Import our new module

# 1. COMMAND LINE ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--start', type=float, default=0.0)
parser.add_argument('--stop', type=float, default=1)
parser.add_argument('--check-geo', action='store_true', help='Visualize geometry and exit without running')
args = parser.parse_args()

# 2. INITIALIZATION
sim = gate.Simulation()
sim.number_of_threads = 1
sim.world.size = [4.0 * gate.g4_units.m] * 3
sim.volume_manager.add_material_database('GateMaterials.db')
sim.random_engine = 'MersenneTwister'
sim.random_seed = args.seed
sim.check_volumes_overlap = False

# 3. BUILD GEOMETRY
build_pet_geometry(sim)
build_test_phantom(sim)

# 4. SETUP PHYSICS, SOURCES, & DIGITIZER
setup.setup_physics(sim)
setup.setup_sources(sim)
setup.setup_digitizer(sim)

# 5. EXECUTION
sim.run_timing_intervals = [[args.start * gate.g4_units.s, args.stop * gate.g4_units.s]]

if args.check_geo:
    print("🔍 Geometry Check Mode: Initializing viewer...")
    
    # 1. Turn on the interactive Qt viewer
    sim.visu = True
    sim.visu_type = 'vrml' 
    
    # 2. Turn ON overlap checking
    sim.check_volumes_overlap = False
    
    # 3. The Trick: Set the time interval from 0 to 0
    sim.run_timing_intervals = [[0.0 * gate.g4_units.s, 0.0 * gate.g4_units.s]]
    
    # 4. Call run(). It will build the geometry, open the GUI, and simulate 0 decays.
    sim.run()
    
    print("Geometry loaded. Close the Qt window to exit the script.")

else:
    # Standard Production Mode
    sim.number_of_threads = args.threads
    sim.visu = False
    sim.run_timing_intervals = [[args.start * gate.g4_units.s, args.stop * gate.g4_units.s]]
    
    print(f"🚀 Running physics simulation with {sim.number_of_threads} threads...")
    sim.run()
