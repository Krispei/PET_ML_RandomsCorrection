'''
    This script controls the simulation for GATE. 
'''
import yaml
import opengate as gate
from sim.build_scanner import build_petcoil_geometry
from sim.phantom import *
import sim.simulation_setup as setup  
import os

#Load parameters
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Units
m = gate.g4_units.m
mm = gate.g4_units.mm
s = gate.g4_units.s
Bq = gate.g4_units.Bq

venv_base = os.path.abspath("GateVenv")
data_dir = os.path.join(venv_base, "share/Geant4/data")

os.environ['G4RADIOACTIVEDATA'] = os.path.join(data_dir, 'RadioactiveDecay5.6')
os.environ['G4ENSDFSTATEDATA'] = os.path.join(data_dir, 'G4ENSDFSTATE2.3')
os.environ['G4LEVELGAMMADATA'] = os.path.join(data_dir, 'PhotonEvaporation5.7')

# initialize
sim = gate.Simulation()
sim.world.size = [4.0 * m] * 3
sim.volume_manager.add_material_database('/Users/wonupark/PET_ML_RandomsCorrection/config/GateMaterials.db')
sim.random_engine = 'MersenneTwister'
sim.random_seed = config["sim"]["seed"]
sim.check_volumes_overlap = False
sim.progress_bar = True

build_petcoil_geometry(sim)

include_f18 = config['phantom']['include_f18']
f18_offset = config['phantom']['f18_offset_mm']
f18_activity = float(config['phantom']['f18_activity_bq'])

include_Mn52 = config['phantom']['include_Mn52']
Mn52_offset = config['phantom']['Mn52_offset_mm']
Mn52_activity = float(config['phantom']['Mn52_activity_bq'])

offsets = []

if include_f18: 
    offsets.append( [dim * mm for dim in  f18_offset] )
if include_Mn52:
    offsets.append( [dim * mm for dim in Mn52_offset] )

phantom_vols = build_nema_nu2_scatter_phantom(sim, sleeve_offsets=offsets)

if include_f18 and include_Mn52:
    add_nema_scatter_source(sim, phantom_vols['line_source_lumen_0'], isotope='F18', activity_bq=f18_activity)
    add_nema_scatter_source(sim, phantom_vols['line_source_lumen_1'], isotope='Mn52', activity_bq=Mn52_activity)
else:
    if include_f18:
        add_nema_scatter_source(sim, phantom_vols['line_source_lumen_0'], isotope='F18', activity_bq=f18_activity)
    if include_Mn52:
        add_nema_scatter_source(sim, phantom_vols['line_source_lumen_0'], isotope='Mn52', activity_bq=Mn52_activity)

energy_window_keV = config['digitizer']['energy_window_keV']
output_filename = config['sim']['output_filename']
setup.setup_physics(sim)
setup.setup_digitizer(sim=sim, energy_window_keV=energy_window_keV, output_filename=output_filename)

start = config['sim']['start']
stop = config['sim']['stop']
check_geo = config['sim']['check_geo']
threads = config['sim']['threads']

if check_geo:
    
    sim.visu = True
    sim.visu_type = 'vrml' 
    sim.check_volumes_overlap = False
    sim.run_timing_intervals = [[0.0 * s, 0.0 * s]]
    sim.run()
    
else:
    sim.number_of_threads = threads
    sim.visu = False
    sim.run_timing_intervals = [[start * s, stop * s]]

    print("=========== PARAMS ===========")
    print(f"Start : {start}")
    print(f"Stop : {stop}")
    print(f"threads : {threads}")
    print("==============================")
    sim.run()


