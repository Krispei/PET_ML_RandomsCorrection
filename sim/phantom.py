
import opengate as gate

def add_na22_point_source_phantom(sim, activity_mbq=20.48):
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    MBq = 1e6 * Bq

    # 1. THE ACRYLIC HOUSING (10x10x10 mm cube)
    # The paper uses this to provide a consistent scatter/attenuation environment 
    cube = sim.add_volume('Box', 'source_housing')
    cube.mother = 'world'
    cube.size = [10.0 * mm, 10.0 * mm, 10.0 * mm]
    cube.material = 'G4_LUCITE'  # Acrylic/Lucite 
    cube.color = [0.5, 0.5, 0.5, 0.5] # Semi-transparent gray
    cube.translation = [0, 0, 0] # Positioned at the center for calibration [cite: 141]
    
    # 2. THE RADIOACTIVE SOURCE
    source = sim.add_source('GenericSource', 'na22_source')
    
    # Using 'ion' ensures OpenGATE handles the 90.6% positron branching ratio [cite: 168]
    # and the 1275 keV prompt gamma emission 
    source.particle = 'ion'
    source.start_time = 0 * gate.g4_units.s
    source.ion.Z = 11  # Sodium
    source.ion.A = 22  # Mass number 22 
    
    source.activity = activity_mbq * MBq
    
    # 100 micrometer diameter point source 
    source.position.type = 'sphere'
    source.position.radius = 0.05 * mm # 50um radius = 100um diameter
    source.position.translation = [0, 0, 0] # Center of the cube
    source.start_time = 0 * gate.g4_units.s

    source.direction.type = 'iso'

    return cube, source

def build_nema_nu2_scatter_phantom(sim, sleeve_offsets=[[0,0,0]]):
    """
    NEMA NU-2 Scatter Phantom (Section 4.0)
    
    Specifications:
    - Outer cylinder: 20 cm diameter, 70 cm long, polyethylene walls (3.2 mm thick)
    - Filled with water
    - Line source: 70 cm active length, running axially through center
    - The phantom is used to measure scatter fraction, count losses, and randoms
    
    Reference: NEMA Standards Publication NU 2-2018
    """
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm

    # -------------------------------------------------------------------------
    # Material Definitions (ensure these exist in your material database)
    # We use G4_POLYETHYLENE for the shell and G4_WATER for the fill.
    # -------------------------------------------------------------------------

    # =========================================================================
    # 1. OUTER POLYETHYLENE SHELL (hollow cylinder)
    #    NEMA spec: 203.2 mm OD, 196.8 mm ID, 700 mm long
    #    Wall thickness = 3.2 mm
    # =========================================================================
    outer_diameter_mm = 203.2   # mm  (~20 cm OD)
    wall_thickness_mm = 3.2     # mm  (NEMA specified)
    inner_diameter_mm = outer_diameter_mm - 2.0 * wall_thickness_mm  # 196.8 mm
    phantom_length_mm = 200.0   # mm  (20 cm axial length)

    phantom_shell = sim.add_volume('Tubs', 'phantom_shell')
    phantom_shell.mother = 'world'
    phantom_shell.rmin = (inner_diameter_mm / 2.0) * mm
    phantom_shell.rmax = (outer_diameter_mm / 2.0) * mm
    phantom_shell.dz   = (phantom_length_mm / 2.0) * mm   # half-length in Geant4
    phantom_shell.sphi = 0.0
    phantom_shell.dphi = 360.0   # degrees → full cylinder
    phantom_shell.material = 'G4_POLYETHYLENE'
    phantom_shell.translation = [0.0, 0.0, 0.0]
    phantom_shell.color = [0.8, 0.8, 0.8, 0.4]   # translucent grey

    # =========================================================================
    # 2. END CAPS (two polyethylene disks, 3.2 mm thick)
    # =========================================================================
    endcap_thickness_mm = 3.2   # mm — matches wall thickness per NEMA

    for side, sign in [('top', +1), ('bottom', -1)]:
        cap = sim.add_volume('Tubs', f'phantom_endcap_{side}')
        cap.mother = 'world'
        cap.rmin = 0.0
        cap.rmax = (outer_diameter_mm / 2.0) * mm
        cap.dz   = (endcap_thickness_mm / 2.0) * mm
        cap.sphi = 0.0
        cap.dphi = 360.0
        cap.material = 'G4_POLYETHYLENE'
        # Position: ± (half phantom length + half cap thickness)
        z_pos = sign * ((phantom_length_mm / 2.0) + (endcap_thickness_mm / 2.0)) * mm
        cap.translation = [0.0, 0.0, z_pos]
        cap.color = [0.6, 0.6, 0.6, 0.6]

    # =========================================================================
    # 3. WATER FILL (inside the shell)
    # =========================================================================
    water_fill = sim.add_volume('Tubs', 'water_fill')
    water_fill.mother = 'world'
    water_fill.rmin = 0.0
    water_fill.rmax = (inner_diameter_mm / 2.0) * mm
    water_fill.dz   = (phantom_length_mm / 2.0) * mm
    water_fill.sphi = 0.0
    water_fill.dphi = 360.0
    water_fill.material = 'G4_WATER'
    water_fill.translation = [0.0, 0.0, 0.0]
    water_fill.color = [0.0, 0.4, 1.0, 0.15]   # very translucent blue
    
    volumes = {}
    
    # Loop to create multiple sleeves
    for i, offset in enumerate(sleeve_offsets):
        sleeve_name = f'line_source_sleeve_{i}'
        lumen_name = f'line_source_lumen_{i}'
        
        # 1. The Sleeve (Polyethylene)
        line_sleeve = sim.add_volume('Tubs', sleeve_name)
        line_sleeve.mother = 'water_fill'
        line_sleeve.rmin = 1.6 * mm # 3.2 ID
        line_sleeve.rmax = 2.4 * mm # 4.8 OD
        line_sleeve.dz = 100.0 * mm # 20cm total length
        line_sleeve.material = 'G4_POLYETHYLENE'
        line_sleeve.translation = offset
        line_sleeve.color = [1.0, 1.0, 0.0, 0.8] # Yellow

        # 2. The Lumen (Water where activity lives)
        line_source_vol = sim.add_volume('Tubs', lumen_name)
        line_source_vol.mother = 'water_fill'
        line_source_vol.rmin = 0.0
        line_source_vol.rmax = 1.6 * mm
        line_source_vol.dz = 100.0 * mm
        line_source_vol.material = 'G4_WATER'
        line_source_vol.translation = offset
        line_source_vol.color = [1.0, 0.0, 0.0, 1.0] # Red
        
        volumes[lumen_name] = line_source_vol

    return volumes

def add_nema_scatter_source(sim, lumen_volume, isotope='F18', activity_bq=50e6):
    source = sim.add_source('GenericSource', f'line_source_{lumen_volume.name}_{isotope}')
    
    source.particle = 'ion'
    source.user_particle_life_time = 0 * gate.g4_units.s
    
    if isotope == 'F18':
        source.ion.Z = 9
        source.ion.A = 18
    elif isotope == 'Mn52':
        source.ion.Z = 25
        source.ion.A = 52
    
    # ---------------------------------------------------------
    # DYNAMIC GEOMETRY MATCHING
    # ---------------------------------------------------------
    source.position.type = 'cylinder'
    source.position.radius = lumen_volume.rmax
    source.position.dz = lumen_volume.dz 
    
    # Use the translation of the specific lumen volume passed in
    source.position.translation = lumen_volume.translation 
    
    source.direction.type = 'iso'
    source.start_time = 0 * gate.g4_units.s
    source.activity = activity_bq * gate.g4_units.Bq

    return source