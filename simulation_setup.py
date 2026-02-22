# simulation_setup.py
import opengate as gate

def setup_physics(sim):
    sim.physics_manager.physics_list_name = 'G4EmStandardPhysics_option4'
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut('world', 'all', 1.0 * gate.g4_units.mm)

import opengate as gate

def setup_sources(sim):
    # ---------------------------------------------------------
    # SOURCE 1: Fluorine-18 (Left Side)
    # ---------------------------------------------------------
    f18 = sim.add_source('GenericSource', 'F18_Source')
    f18.particle = 'ion 9 18' 
    f18.energy.mono = 0.0 * gate.g4_units.MeV
    f18.activity = 200e6 * gate.g4_units.Bq 
    f18.start_time = 0 * gate.g4_units.s
    f18.end_time   = 0.01 * gate.g4_units.s

    # Define spatial shape and location
    f18.position.type = 'sphere'
    f18.position.radius = 10.0 * gate.g4_units.mm
    # Move 50 mm along the negative X-axis
    f18.position.translation = [-50.0 * gate.g4_units.mm, 0.0, 0.0] 

    # ---------------------------------------------------------
    # SOURCE 2: Mn52 (Right Side)
    # ---------------------------------------------------------

    Mn52 = sim.add_source('GenericSource', 'Mn52_Source')
    Mn52.particle = 'ion 9 18'
    Mn52.energy.mono = 0.0 * gate.g4_units.MeV
    Mn52.activity = 200e6 * gate.g4_units.Bq
    
    # Define spatial shape and location
    Mn52.position.type = 'sphere'
    Mn52.position.radius = 10.0 * gate.g4_units.mm
    # Move 50 mm along the positive X-axis
    Mn52.position.translation = [50.0 * gate.g4_units.mm, 0.0, 0.0]
    Mn52.start_time = 0 * gate.g4_units.s
    Mn52.end_time = 0.01 * gate.g4_units.s

def setup_digitizer(sim):
    # 1. Raw Hits
    hits = sim.add_actor('DigitizerHitsCollectionActor', 'hits')
    hits.attached_to = 'LYSO'
    hits.authorize_repeated_volumes = True
    hits.attributes = ['TotalEnergyDeposit', 'PostPosition', 'GlobalTime', 
                       'EventID', 'EventPosition', 'PreStepUniqueVolumeID']

    # 2. Adder
    adder = sim.add_actor('DigitizerAdderActor', 'Singles')
    adder.attached_to = 'LYSO'
    adder.input_digi_collection = 'hits'
    adder.policy = 'EnergyWeightedCentroidPosition'
    adder.authorize_repeated_volumes = True

    # 3. Blurring
    blur = sim.add_actor('DigitizerBlurringActor', 'BlurredSingles')
    blur.attached_to = 'LYSO'
    blur.input_digi_collection = 'Singles'
    blur.blur_attribute = 'TotalEnergyDeposit' 
    blur.blur_method = 'InverseSquare' 
    blur.blur_resolution = 0.116
    blur.blur_reference_value = 511.0 * gate.g4_units.keV
    blur.authorize_repeated_volumes = True
    
    time_blur = sim.add_actor('DigitizerBlurringActor', 'TimeBlur')
    time_blur.input_digi_collection = 'BlurredSingles'
    time_blur.blur_attribute = 'GlobalTime'
    time_blur.blur_method = 'Gaussian'
    time_blur.blur_sigma = 85 * gate.g4_units.ps 

    # 4. Energy Window & Output
    window = sim.add_actor('DigitizerEnergyWindowsActor', 'FilteredSingles')
    window.input_digi_collection = 'TimeBlur'
    window.channels = [{'name': 'photopeak', 'min': 450.0 * gate.g4_units.keV, 
                        'max': 600.0 * gate.g4_units.keV}]
    window.output_filename = 'test_singles.root'
