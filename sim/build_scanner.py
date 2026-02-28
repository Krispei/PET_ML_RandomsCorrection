import opengate as gate

def build_petcoil_geometry(sim):
    mm = gate.g4_units.mm

    # 1. CYLINDRICAL ENVELOPE (Matches 315mm ID, 432mm OD)
    cyl = sim.add_volume('Tubs', 'cylindricalPET')
    cyl.mother = 'world'
    cyl.rmin = 157.5 * mm  # ID/2 = 315/2
    cyl.rmax = 216.0 * mm  # OD/2 = 432/2
    cyl.dz = 101.0 * mm    # Faraday cage length is 202mm; dz is half-length
    cyl.material = 'G4_AIR'

    # 2. RSECTOR (16 modules)
    rsector = sim.add_volume('Box', 'rsector')
    rsector.mother = 'cylindricalPET'
    rsector.size = [58.0 * mm, 58.0 * mm, 202.0 * mm] # Based on Table 1 
    rsector.material = 'G4_AIR'

    # Rotation for 16 modules with 4.6mm gaps [cite: 116, 117]
    # Radius must account for the 58mm module thickness to stay within ID/OD
    trans_ring, rot_ring = gate.geometry.utility.get_circular_repetition(
        number_of_repetitions=16, 
        first_translation=[186.75 * mm, 0.0, 0.0], # (315/2) + (58/2)
        axis=[0, 0, 1]
    )
    rsector.translation = trans_ring
    rsector.rotation = rot_ring

    # 3. MODULE (Inside Faraday Cage)
    module = sim.add_volume('Box', 'module')
    module.mother = 'rsector'
    module.size = [58.0 * mm, 58.0 * mm, 202.0 * mm]
    module.material = 'G4_AIR'

    # 4. SUBMODULE (6 sub-modules arranged axially) [cite: 119, 123]
    submodule = sim.add_volume('Box', 'submodule')
    submodule.mother = 'module'
    # Each sub-module houses an 8x16 crystal array
    submodule.size = [20.0 * mm, 52.16 * mm, 26.08 * mm] 
    submodule.material = 'G4_AIR'

    # Axial repetition of 6 submodules
    sub_translations = gate.geometry.utility.get_grid_repetition(
        size=[1, 1, 6], 
        spacing=[0.0, 0.0, 26.7 * mm] # Approx pitch to fit 160mm FOV [cite: 118]
    )
    submodule.translation = sub_translations

    # 5. CRYSTAL (LYSO 3.2 x 3.2 x 20 mm) 
    crystal = sim.add_volume('Box', 'LYSO')
    crystal.mother = 'submodule'
    crystal.size = [20.0 * mm, 3.19 * mm, 3.19 * mm]
    crystal.material = 'LYSO'

    # Grid: 16 transaxial (Y) x 8 axial (Z) [cite: 123]
    cryst_translations = gate.geometry.utility.get_grid_repetition(
        size=[1, 16, 8], 
        spacing=[0.0, 3.26 * mm, 3.26 * mm] # Pitch from Table 1 
    )
    crystal.translation = cryst_translations

    return crystal