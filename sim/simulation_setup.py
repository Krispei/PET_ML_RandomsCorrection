'''
    This code contains the detector settings as well as the physics settings
'''

import opengate as gate

def setup_physics(sim):
    sim.physics_manager.physics_list_name = 'G4EmStandardPhysics_option4'
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut('world', 'all', 1.0 * gate.g4_units.mm)

def setup_digitizer(sim, energy_window_keV, output_filename):
    hits = sim.add_actor('DigitizerHitsCollectionActor', 'hits')
    hits.attached_to = 'LYSO'
    hits.authorize_repeated_volumes = True
    hits.attributes = ['TotalEnergyDeposit', 'PostPosition', 'GlobalTime', 
                       'EventID', 'EventPosition', 'PreStepUniqueVolumeID']

    adder = sim.add_actor('DigitizerAdderActor', 'Singles')
    adder.attached_to = 'LYSO'
    adder.input_digi_collection = 'hits'
    adder.policy = 'EnergyWeightedCentroidPosition'
    adder.authorize_repeated_volumes = True

    blur = sim.add_actor('DigitizerBlurringActor', 'BlurredSingles')
    blur.attached_to = 'LYSO'
    blur.input_digi_collection = 'Singles'
    blur.blur_attribute = 'TotalEnergyDeposit' 
    blur.blur_method = 'InverseSquare' 
    blur.blur_resolution = 0.1174
    blur.blur_reference_value = 511.0 * gate.g4_units.keV
    blur.authorize_repeated_volumes = True
    
    time_blur = sim.add_actor('DigitizerBlurringActor', 'TimeBlur')
    time_blur.input_digi_collection = 'BlurredSingles'
    time_blur.blur_attribute = 'GlobalTime'
    time_blur.blur_method = 'Gaussian'
    time_blur.blur_sigma = 0.071 * gate.g4_units.ns

    window = sim.add_actor('DigitizerEnergyWindowsActor', 'FilteredSingles')
    window.input_digi_collection = 'TimeBlur'
    window.channels = [{'name': 'photopeak', 'min': energy_window_keV[0] * gate.g4_units.keV, 
                        'max': energy_window_keV[1] * gate.g4_units.keV}]
    window.output_filename = f'{output_filename}.root'

    stats = sim.add_actor('SimulationStatisticsActor', 'stats')
    stats.output_filename = 'stats.txt'