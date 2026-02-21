# simulation_setup.py
import opengate as gate

def setup_physics(sim):
    sim.physics_manager.physics_list_name = 'G4EmStandardPhysics_option4'
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut('world', 'all', 1.0 * gate.g4_units.mm)

def setup_sources(sim):
    # Setup the validation point source
    test_src = sim.add_source('GenericSource', 'test_source')
    test_src.particle = 'e+'
    test_src.energy.mono = 0.0 * gate.g4_units.MeV
    test_src.activity = 1e6 * gate.g4_units.Bq
    test_src.attached_to = 'source_drop'

def setup_digitizer(sim):
    # 1. Raw Hits
    hits = sim.add_actor('DigitizerHitsCollectionActor', 'hits')
    hits.attached_to = 'LYSO'
    hits.authorize_repeated_volumes = True
    hits.attributes = ['TotalEnergyDeposit', 'PostPosition', 'GlobalTime', 
                       'EventID', 'PreStepUniqueVolumeID']

    # 2. Adder
    adder = sim.add_actor('DigitizerAdderActor', 'Singles')
    adder.input_digi_collection = 'hits'
    adder.policy = 'EnergyWeightedCentroidPosition'

    # 3. Blurring
    blur = sim.add_actor('DigitizerBlurringActor', 'BlurredSingles')
    blur.input_digi_collection = 'Singles'
    blur.blur_attribute = 'TotalEnergyDeposit' 
    blur.blur_method = 'InverseSquare' 
    blur.blur_resolution = 0.116
    blur.blur_reference_value = 511.0 * gate.g4_units.keV

    # 4. Energy Window & Output
    window = sim.add_actor('DigitizerEnergyWindowsActor', 'FilteredSingles')
    window.input_digi_collection = 'BlurredSingles'
    window.channels = [{'name': 'photopeak', 'min': 450.0 * gate.g4_units.keV, 
                        'max': 600.0 * gate.g4_units.keV}]
    window.output_filename = 'test_singles.root'