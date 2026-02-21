import opengate as gate

def build_test_phantom(sim):
    mm = gate.g4_units.mm

    # 1. Scattering Sphere (Physics Validation)
    water_sphere = sim.add_volume('Sphere', 'water_sphere')
    water_sphere.mother = 'world'
    water_sphere.rmax = 50.0 * mm  # 50 mm radius = 10 cm diameter
    water_sphere.rmin = 0.0 * mm
    water_sphere.material = 'G4_WATER'
    water_sphere.color = [0, 0, 1, 0.3]  # Translucent blue

    # 2. Source Container (Geometry Validation)
    source_drop = sim.add_volume('Sphere', 'source_drop')
    source_drop.mother = 'water_sphere'
    source_drop.rmax = 1.0 * mm  # Tiny 1 mm radius drop
    source_drop.translation = [0.0, 0.0, 0.0] # Exact center
    source_drop.material = 'G4_WATER'
    source_drop.color = [1, 0, 0, 1]  # Solid red

    # 1. Visible Sphere for F-18 (Left Side)
    f18_vol = sim.add_volume('Sphere', 'F18_Volume')
    f18_vol.mother = 'world'  # Or whatever your main phantom container is
    f18_vol.material = 'G4_WATER'
    f18_vol.rmax = 10.0 * gate.g4_units.mm
    f18_vol.translation = [-50.0 * gate.g4_units.mm, 0.0, 0.0]
    # Set to a highly visible semi-transparent Red
    f18_vol.color = [1.0, 0.0, 0.0, 0.5] 

    # 2. Visible Sphere for Ga-68 (Right Side)
    ga68_vol = sim.add_volume('Sphere', 'Ga68_Volume')
    ga68_vol.mother = 'world'
    ga68_vol.material = 'G4_WATER'
    ga68_vol.rmax = 10.0 * gate.g4_units.mm
    ga68_vol.translation = [50.0 * gate.g4_units.mm, 0.0, 0.0]
    # Set to a highly visible semi-transparent Green
    ga68_vol.color = [0.0, 1.0, 0.0, 0.5]