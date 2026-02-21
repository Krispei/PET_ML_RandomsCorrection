# my_scanner.py
import opengate as gate

def build_pet_geometry(sim):

    mm = gate.g4_units.mm

    """Builds the 16-block cylindrical PET scanner geometry."""
    
    # 1. CYLINDRICAL ENVELOPE
    cyl = sim.add_volume('Tubs', 'cylindricalPET')
    cyl.mother = 'world'
    cyl.rmax = 220.0 * mm
    cyl.rmin = 140.0 * mm
    cyl.dz = 110.0 * mm 
    cyl.material = 'G4_AIR'
    cyl.color = [1, 1, 1, 1] 

    # 2. RSECTOR
    rsector = sim.add_volume('Box', 'rsector')
    rsector.mother = 'cylindricalPET'
    rsector.size = [80.0*mm, 80.0*mm, 220.0*mm] 
    rsector.material = 'G4_AIR'
    rsector.color = [1, 1, 0, 1]  #Yellow

    trans_ring, rot_ring = gate.geometry.utility.get_circular_repetition(
        number_of_repetitions=16, 
        first_translation=[172.6 * mm, 0.0, 0.0], 
        axis=[0, 0, 1]
    )
    rsector.translation = trans_ring
    rsector.rotation = rot_ring

    # 3. MODULE
    module = sim.add_volume('Box', 'module')
    module.mother = 'rsector'
    module.size = [58.5*mm, 58.0*mm, 201.2*mm]
    module.material = 'G4_AIR'
    module.color = [1, 1, 0, 1]

    # 4. SUBMODULE
    submodule = sim.add_volume('Box', 'submodule')
    submodule.mother = 'module'
    submodule.size = [21.0*mm, 53.2*mm, 26.2*mm]
    submodule.material = 'G4_AIR'
    submodule.color = [1, 0, 0, 1]

    sub_translations = gate.geometry.utility.get_grid_repetition(
        size=[1, 1, 6], 
        spacing=[0.0, 0.0, 26.7 * mm]
    )
    submodule.translation = sub_translations

    # 5. CRYSTAL
    crystal = sim.add_volume('Box', 'crystal')
    crystal.mother = 'submodule'
    crystal.size = [20.0*mm, 3.19*mm, 3.19*mm]
    crystal.material = 'G4_AIR'
    crystal.color = [0, 0, 1, 1] 

    cryst_translations = gate.geometry.utility.get_grid_repetition(
        size=[1, 16, 8], 
        spacing=[0.0, 3.26*mm, 3.26*mm]
    )
    crystal.translation = cryst_translations

    # 6. LYSO
    lyso = sim.add_volume('Box', 'LYSO')
    lyso.mother = 'crystal'
    lyso.size = [20.0*mm, 3.19*mm, 3.19*mm]
    lyso.material = 'LYSO'
    lyso.color = [0, 0, 1, 1]