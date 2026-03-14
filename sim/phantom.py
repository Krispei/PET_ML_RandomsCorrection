
import opengate as gate

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

    outer_diameter_mm = 203.2
    wall_thickness_mm = 3.2  
    inner_diameter_mm = outer_diameter_mm - 2.0 * wall_thickness_mm
    phantom_length_mm = 200.0 

    phantom_shell = sim.add_volume('Tubs', 'phantom_shell')
    phantom_shell.mother = 'world'
    phantom_shell.rmin = (inner_diameter_mm / 2.0) * mm
    phantom_shell.rmax = (outer_diameter_mm / 2.0) * mm
    phantom_shell.dz   = (phantom_length_mm / 2.0) * mm 
    phantom_shell.sphi = 0.0
    phantom_shell.dphi = 360.0  
    phantom_shell.material = 'G4_POLYETHYLENE'
    phantom_shell.translation = [0.0, 0.0, 0.0]
    phantom_shell.color = [0.8, 0.8, 0.8, 0.4]  


    endcap_thickness_mm = 3.2 
    for side, sign in [('top', +1), ('bottom', -1)]:
        cap = sim.add_volume('Tubs', f'phantom_endcap_{side}')
        cap.mother = 'world'
        cap.rmin = 0.0
        cap.rmax = (outer_diameter_mm / 2.0) * mm
        cap.dz   = (endcap_thickness_mm / 2.0) * mm
        cap.sphi = 0.0
        cap.dphi = 360.0
        cap.material = 'G4_POLYETHYLENE'
        z_pos = sign * ((phantom_length_mm / 2.0) + (endcap_thickness_mm / 2.0)) * mm
        cap.translation = [0.0, 0.0, z_pos]
        cap.color = [0.6, 0.6, 0.6, 0.6]

    water_fill = sim.add_volume('Tubs', 'water_fill')
    water_fill.mother = 'world'
    water_fill.rmin = 0.0
    water_fill.rmax = (inner_diameter_mm / 2.0) * mm
    water_fill.dz   = (phantom_length_mm / 2.0) * mm
    water_fill.sphi = 0.0
    water_fill.dphi = 360.0
    water_fill.material = 'G4_WATER'
    water_fill.translation = [0.0, 0.0, 0.0]
    water_fill.color = [0.0, 0.4, 1.0, 0.15] 
    
    volumes = {}
    
    for i, offset in enumerate(sleeve_offsets):
        sleeve_name = f'line_source_sleeve_{i}'
        lumen_name = f'line_source_lumen_{i}'
        
        line_sleeve = sim.add_volume('Tubs', sleeve_name)
        line_sleeve.mother = 'water_fill'
        line_sleeve.rmin = 1.6 * mm 
        line_sleeve.rmax = 2.4 * mm 
        line_sleeve.dz = 100.0 * mm 
        line_sleeve.material = 'G4_POLYETHYLENE'
        line_sleeve.translation = offset
        line_sleeve.color = [1.0, 1.0, 0.0, 0.8]

        line_source_vol = sim.add_volume('Tubs', lumen_name)
        line_source_vol.mother = 'water_fill'
        line_source_vol.rmin = 0.0
        line_source_vol.rmax = 1.6 * mm
        line_source_vol.dz = 100.0 * mm
        line_source_vol.material = 'G4_WATER'
        line_source_vol.translation = offset
        line_source_vol.color = [1.0, 0.0, 0.0, 1.0]
    
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
    
    source.position.type = 'cylinder'
    source.position.radius = lumen_volume.rmax
    source.position.dz = lumen_volume.dz 
    source.position.translation = lumen_volume.translation 
    source.direction.type = 'iso'
    source.start_time = 0 * gate.g4_units.s
    source.activity = activity_bq * gate.g4_units.Bq

    return source