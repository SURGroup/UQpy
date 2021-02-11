## Audrey Olivier, March 2020
## Utility functions for LS-DYNA model

import numpy as np
from itertools import islice
rve_size = (0.1, 0.1, 0.05)    # dimension of rectangle in mm


################################################################################
# write to the .k file
################################################################################

def write_geometry_voids(kfile, nvoids):
    with open(kfile, 'a') as f:
        f.write('*INITIAL_VOLUME_FRACTION_GEOMETRY\n')
        f.write('$ Base filling = base material\n')
        f.write('$      sid     idtyp     bammg    ntrace\n')
        #f.write('{:>10d}{:>10d}{:>10d}\n'.format(1, 1, 1))
        f.write('{:>10d}{:>10d}{:>10d}\n'.format(1, 1, 2))
        f.write('$ Filling action 1: create voids\n')
        f.write('$     type   fillopt     fammg\n')
        f.write('$       xi        yi        zi        ri\n')
        for i in range(nvoids):
            #f.write('{:>10d}{:>10d}{:>10d}\n'.format(6, 0, 2))
            f.write('{:>10d}{:>10d}{:>10d}\n'.format(6, 0, 1))
            f.write('<x[{}]><y[{}]><z[{}]><R[{}]>\n'.format(i, i, i, i))
        f.write('$ Filling action 2: create void outside a BOX\n')
        f.write('{:>10d}{:>10d}{:>10d}\n'.format(5, 1, 3))
        f.write('{:>10f}{:>10f}{:>10f}{:>10f}{:>10f}{:>10f}\n'.format(0., 0., 0., 0.1, 0.1, 0.05))


def write_BCs(kfile, V_x, V_y, V_z, ramp_up_velocity=True, t1=1e-4):
    # Set the final time, slighlty larger if velocity is ramped up to get to a final displacement of 0.045 mm
    end_time = compute_tf(t1=t1, V=V_x, ramp_up_velocity=ramp_up_velocity)
    if ramp_up_velocity:
        lcid = 1
        with open(kfile, 'a') as f:
            f.write('$ Define curves needed for setting BCs\n')
            f.write('*DEFINE_CURVE\n')
            f.write('$     LCID      SIDR       SFA       SFO\n')
            f.write('         1         0         1         1\n')
            f.write('                 0.0                 0.0\n')
            f.write('          {:>10f}                 1.0\n'.format(t1))
            f.write('          {:>10f}                 1.0\n'.format(end_time))
    else:
        lcid = 2
        with open(kfile, 'a') as f:
            f.write('$ Define curves needed for setting BCs\n')
            f.write('*DEFINE_CURVE\n')
            f.write('$     LCID      SIDR       SFA       SFO\n')
            f.write('         2         0         1         1\n')
            f.write('                 0.0                 1.0\n')
            f.write('          {:>10f}                 1.0\n'.format(end_time))
            f.write('*INCLUDE\n')
            if V_y is None and V_z is None and V_x == 100.:    # uniaxial case
                f.write('init_vel_ale_1comp.k\n')
                f.write('*INITIAL_VELOCITY_RIGID_BODY\n')
                f.write('         5{:>10f}\n'.format(V_x))
            elif V_y == -20. and V_z == -10. and V_x == 100.:    # triaxial case
                f.write('init_vel_ale_3comp.k\n')
                f.write('*INITIAL_VELOCITY_RIGID_BODY\n')
                f.write('         5{:>10f}\n'.format(V_x))
                f.write('*INITIAL_VELOCITY_RIGID_BODY\n')
                f.write('         7{:>10f}{:>10f}\n'.format(0., V_y))
                f.write('*INITIAL_VELOCITY_RIGID_BODY\n')
                f.write('         9{:>10f}{:>10f}{:>10f}\n'.format(0., 0., V_z))
            else:
                raise ValueError('For initial velocity, velocities are fixed - or you must recreate the '
                                 'init_vel_ale_1comp files, see preprocess_utils_nstes_velocity.py')
    # Write command to apply velocity to the plates and set the end time of calculation
    with open(kfile, 'a') as f:
        f.write('*BOUNDARY_PRESCRIBED_MOTION_RIGID\n')
        f.write('$NID|NSID|PID    DOF       VAD      LCID        SF\n')
        f.write('{:>10d}{:>10d}{:>10d}{:>10d}{:>10f}\n'.format(5, 1, 0, lcid, V_x))
        if V_y is not None:
            f.write('{:>10d}{:>10d}{:>10d}{:>10d}{:>10f}\n'.format(7, 2, 0, lcid, V_y))
        if V_z is not None:
            f.write('{:>10d}{:>10d}{:>10d}{:>10d}{:>10f}\n'.format(9, 3, 0, lcid, V_z))
        f.write('*CONTROL_TERMINATION\n')
        f.write('$      end         min\n')
        f.write('{:>10f}               5e-10\n'.format(end_time))


################################################################################
# Utility functions to sample voids in a specific fashion
################################################################################
def comp_volumes(voids_radii):
    """ Compute volume of voids. """
    return 4. / 3. * np.pi * voids_radii ** 3


def radius_from_volume(volume):
    return (3. / (4. * np.pi) * volume) ** (1/3)


def comp_porosity(voids_radii):
    """ Compute porosity value. """
    volumes = comp_volumes(voids_radii)
    return np.sum(volumes) / np.prod(rve_size)


def sample_voids_radii(n_voids, radii_bounds, porosity_value=None, porosity_bounds=None):
    """ Sample n_voids with radii between radii_bounds
    3 cases: no porosity specified, just sample voids uniformly from radii_bounds
    porosity value specified: porosity must be exactly equal to given value
    porosity bounds: porosity must lie within bounds
    
    radii_bounds and rve_size are given in microns """

    # check inputs
    if porosity_value is not None and porosity_bounds is not None:
        raise ValueError('Cannot specify both porosity_value or porosity_bounds')
    # case 0:
    if porosity_value is None and porosity_bounds is None:
        voids_radii = np.random.uniform(low=radii_bounds[0], high=radii_bounds[1], size=n_voids)
    # case 1
    elif porosity_bounds is not None:
        # sample radii until you get into the right porosity bounds
        nc = 0
        voids_radii = np.random.uniform(low=radii_bounds[0], high=radii_bounds[1], size=n_voids)
        porosity = comp_porosity(voids_radii)
        while nc < 5000 and (porosity > porosity_bounds[1] or porosity < porosity_bounds[0]):
            nc += 1
            voids_radii = np.random.uniform(low=radii_bounds[0], high=radii_bounds[1], size=n_voids)
            porosity = comp_porosity(voids_radii)
        if nc == 5000:
            print('Max. number of iterations reached !')
            voids_radii = None
    elif porosity_value is not None:
        # sample radii until you get into the right porosity value (first sample n-1 voids within acceptable bounds then add the last one)
        volume_bounds = porosity_value * np.prod(rve_size) - comp_volumes(np.array(radii_bounds))
        nc = 0
        voids_radii = np.random.uniform(low=radii_bounds[0], high=radii_bounds[1], size=n_voids - 1)
        volume_voids = np.sum(comp_volumes(voids_radii))
        while nc < 5000 and (volume_voids > volume_bounds[0] or volume_voids < volume_bounds[1]):
            nc += 1
            voids_radii = np.random.uniform(low=radii_bounds[0], high=radii_bounds[1], size=n_voids - 1)
            volume_voids = np.sum(comp_volumes(voids_radii))
        if nc == 5000:
            print('Max. number of iterations reached !')
            voids_radii = None
        else:
            last_radius = radius_from_volume(porosity_value * np.prod(rve_size) - volume_voids)
            voids_radii = np.concatenate([voids_radii, np.array(last_radius).reshape((-1, ))], axis=0)
    return voids_radii


def place_voids_no_overlap(voids_radii, bounds_placement):
    """ Place voids within the domain
    Voids cannot overlap, minimum distance of one radius between a void and its neighbors
    Strategy: place the voids in order of decreasing radius
    Either place all the voids (their centers) within the box defined by bounds placement, or, if bounds placement
    is None, place all the voids at a minimum distance of R/2 from the rve border"""
    if bounds_placement is None:
        bounds_placement = ((0., rve_size[0]), (0., rve_size[1]), (0., rve_size[2]))
    sorted_radii = np.sort(voids_radii)[::-1]
    # place the largest void, just check with itself
    placed_xyz = np.array(
        [np.random.uniform(low=bnds[0], high=bnds[1], size=1) for bnds in bounds_placement]
    ).reshape((1, 3))
    while not check_with_borders(placed_xyz, sorted_radii[0]):
        placed_xyz = np.array(
            [np.random.uniform(low=bnds[0], high=bnds[1], size=1) for bnds in bounds_placement]
        ).reshape((1, 3))
    placed_radii = [sorted_radii[0], ]
    for n, radius in enumerate(sorted_radii[1:]):
        nc = 0
        try_position = np.array(
            [np.random.uniform(low=bnds[0], high=bnds[1], size=1) for bnds in bounds_placement]
        ).reshape((1, 3))
        while nc < 1000 * (n+1) and (
                (not check_with_existing_voids(try_position, radius, placed_xyz, placed_radii)) or
                (not check_with_borders(try_position, radius))):
            nc += 1
            try_position = np.array(
                [np.random.uniform(low=bnds[0], high=bnds[1], size=1) for bnds in bounds_placement]
            ).reshape((1, 3))
        if nc == 1000 * (n+1):
            print('Max. number of iterations reached !')
            placed_xyz = None
            break
        placed_xyz = np.concatenate([placed_xyz, try_position], axis=0)
        placed_radii.append(radius)
    return placed_xyz, placed_radii


def check_with_existing_voids(new_xyz, new_radius, existing_xyz, existing_radii):
    """ Check that the new position we want to add is ok, check with all existing  """
    from sklearn.metrics.pairwise import euclidean_distances
    dist_to_others = euclidean_distances(new_xyz, existing_xyz).reshape((-1, 1))
    if any(d <= max(r, new_radius) + r + new_radius for d, r in zip(dist_to_others, existing_radii)):
        return False
    return True


def check_with_borders(new_xyz, new_radius):
    """ Check that the new position we want to add is ok, ie, at a min distance of R/2 with all borders """
    # Check near 0
    if any(coord <= new_radius + new_radius / 2 for coord in new_xyz.reshape((-1, ))):
        return False
    # Check near top
    if any(top_coord - coord <= new_radius + new_radius / 2 for top_coord, coord
           in zip(rve_size, new_xyz.reshape((-1,)))):
        return False
    return True


################################################################################
# Utility functions to postprocess outputs
################################################################################

def read_alematvol_file(file_name, solid_is_first=False):
    t = []
    volume_mat1 = []
    volume_mat2 = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # data for material 1
    flag_mat1, flag_mat2 = False, False
    for i, x in enumerate(lines):
        if '1#pts=1' in x:
            flag_mat1 = True
            continue
        if '2#pts=1' in x:
            flag_mat2 = True
            continue
        if flag_mat1:
            if 'endcurve' in x:
                flag_mat1 = False
                continue
            time, v = x.split()
            t.append(float(time))
            volume_mat1.append(float(v))
        if flag_mat2:
            if 'endcurve' in x:
                flag_mat2 = False
                break
            time, v = x.split()
            #t.append(float(time))
            volume_mat2.append(float(v))
    if len(volume_mat1) != len(volume_mat2):
        print('Error in post-processing')
    # Always return: time, volume solid, volume voids
    if solid_is_first:
        return np.array(t), np.array(volume_mat1), np.array(volume_mat2)
    return np.array(t), np.array(volume_mat2), np.array(volume_mat1)


def read_elout_file(file_name, scale_to_MPa=1., average_over_solid=True):
    t, stresses = [], []
    tmp_t, tmp_stresses = [], []
    nb_of_lines = 100000
    with open(file_name, 'r') as filehandle:
        while True:
            lines_cache = list(islice(filehandle, nb_of_lines))
            if not lines_cache:
                break
            #print(len(lines_cache))
            for line in lines_cache:
                if 'elementstresscalculationsfortimestep' in line.replace(" ", ""):
                    t.append(tmp_t)
                    stresses.append(tmp_stresses)
                    tmp_t = float(line.split('at time')[1].split(')')[0])
                    tmp_stresses = []
                    continue
                try:
                    assert len(line.split()) == 9
                    sigmas = [float(w) for w in line.split()[1:7]]
                except:
                    continue
                else:
                    tmp_stresses.append(sigmas)
    # add the last computed value
    t.append(tmp_t)
    stresses.append(tmp_stresses)
    # average over all elements
    if not average_over_solid:
        return np.array(t[1:]), np.array(stresses[1:]) * scale_to_MPa
    av_pressure = []
    for stress in np.array(stresses[1:]):
        mask = (stress != 0.)
        mask = (np.sum(mask, axis=1) != 0.)
        if sum(mask) == 0 or sum(mask) == 1:
            av_pressure.append(np.array([0., 0., 0., 0., 0., 0.]))
        else:
            av_pressure.append(np.mean(stress[mask, :], axis=0))
    return np.array(t[1:]), np.array(av_pressure) * scale_to_MPa


def read_bndout_file(file_name, scale_to_N=1., areas=(0.12 * 0.07, 0.16 * 0.07, 0.16 * 0.12)):
    t, forces = [], []
    tmp_t, tmp_forces = [], []
    nb_of_lines = 100000
    # areas are in mm, for parts 5, 7, 9 respectively
    print('areas = {}, {}, {} mm2'.format(areas[0], areas[1], areas[2]))
    with open(file_name, 'r') as filehandle:
        while True:
            lines_cache = list(islice(filehandle, nb_of_lines))
            if not lines_cache:
                break
            #print(len(lines_cache))
            for line in lines_cache:
                if 'nodalforce/energyoutput' in line.replace(" ", ""):
                    t.append(tmp_t)
                    forces.append(tmp_forces)
                    tmp_t = float(line.split('=')[1])
                    tmp_forces = [0., 0., 0.]
                    continue
                try:
                    assert 'mat#' in line.split()[0]
                    assert int(line.split()[1]) in [5, 7, 9]
                except:
                    continue
                else:
                    if int(line.split()[1]) == 5:
                        tmp_forces[0] = float(line.split()[3])
                    if int(line.split()[1]) == 7:
                        tmp_forces[1] = float(line.split()[5])
                    if int(line.split()[1]) == 9:
                        tmp_forces[2] = float(line.split()[7])
    t.append(tmp_t)
    forces.append(tmp_forces)
    return np.array(t[1:]), np.array(forces[1:]) * scale_to_N


def read_dbfsi_file(file_name, scale_to_MPa=1., scale_to_N=1.):
    t, pressure, forces = [], [], []
    nb_of_lines = 100000
    with open(file_name, 'r') as filehandle:
        while True:
            lines_cache = list(islice(filehandle, nb_of_lines))
            if not lines_cache:
                break
            #print(len(lines_cache))
            for i, line in enumerate(lines_cache):
                if 'time=' in line.replace(" ", ""):
                    tmp_t = float(line.split('=')[1])
                    t.append(tmp_t)
                    continue
                try:
                    assert int(line.split()[0]) == 25
                    tmp_line = [float(f) for f in line.split()]
                except:
                    continue
                else:
                    pressure.append(tmp_line[1])
                    forces.append(tmp_line[2:5])
    return np.array(t), np.array(pressure) * scale_to_MPa, np.array(forces) * scale_to_N


def read_spcforc_file(file_name, scale_to_N=1.):
    t, forces = [], []
    nb_of_lines = 100000
    with open(file_name, 'r') as filehandle:
        while True:
            lines_cache = list(islice(filehandle, nb_of_lines))
            if not lines_cache:
                break
            #print(len(lines_cache))
            for line in lines_cache:
                if 'outputattime=' in line.replace(" ", ""):
                    tmp_t = float(line.split('=')[1])
                    t.append(tmp_t)
                    continue
                if 'forceresultants=' in line.replace(" ", ""):
                    tmp_forces = [float(f) for f in line.split('=')[1].split()]
                    forces.append(tmp_forces)
                    continue
    return np.array(t), np.array(forces) * scale_to_N


# Utility functions: compute displacements from velocity curves
def compute_displacement_curve(tvec, V=100., ramp_up_velocity=True, t1=1e-4):
    """ Velocity is either ramped up from 0 to V between t=0 and t=t1, then constant V(t>t1)=V, or just constant """
    dx = np.zeros_like(tvec)
    if ramp_up_velocity:
        for i, t in enumerate(tvec):
            if t <= t1:
                dx[i] = 0.5 * V / t1 * t ** 2
            else:
                dx[i] = V * (t - 0.5 * t1)
    else:
        for i, t in enumerate(tvec):
            dx[i] = V * t
    return dx


def compute_tf(V=100., ramp_up_velocity=True, t1=1e-4):
    # Compute final time, for an overall displacement of 0.045 mm
    if ramp_up_velocity:
        tf = t1 / 2 + 0.045 / V
    else:
        tf = 0.045 / V
    return tf
