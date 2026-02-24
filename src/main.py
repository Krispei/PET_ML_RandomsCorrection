# main.py
import opengate as gate
import argparse
from build_scanner import build_pet_geometry
from NEMA_NU2_phantom import build_test_phantom
import simulation_setup as setup 

# Units

m = gate.g4_units.m # Meters
s = gate.g4_units.s # Seconds

# 1. COMMAND LINE ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--start', type=float, default=0.0)
parser.add_argument('--stop', type=float, default=1)
args = parser.parse_args()

# 2. INITIALIZATION
sim = gate.Simulation()
sim.number_of_threads = 8
sim.world.size = [4.0 * m] * 3
sim.volume_manager.add_material_database('/Users/wonupark/PET_ML_RandomsCorrection/config/GateMaterials.db')
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
sim.run_timing_intervals = [[args.start * s, args.stop * s]]

# --- VISUALIZATION ---
sim.visu = False
sim.visu_type = 'vrml' 

sim.run()
