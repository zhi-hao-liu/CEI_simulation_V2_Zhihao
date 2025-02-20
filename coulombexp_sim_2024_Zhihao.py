"""
Coulomb Explosion Simulation Program (Optimized & Restructured)
==============================================================

Author:         Zhihao Liu
Contact:        siclinliu [AT] gmail.com
Adapted by:     (You / Your Team)
Last Modified:  2025-02-18

Description
-----------
This module provides the core functionality for simulating a
Coulomb explosion experiment:
1. Reading geometry and vibrational information from Gaussian files.
2. Constructing data classes for atoms, molecules, and ionic fragments.
3. Performing vibrational sampling.
4. Solving classical equations of motion for the charged fragments
   generated in the explosion.
5. Returning final positions, velocities, etc. for analysis.

Main Classes
------------
- Experiment
- atom
- molecule
- fragment

Usage
-----
In your main script or Jupyter notebook, import this module:

    from coulombexp_sim_2021_routines import *

Then create an Experiment object from an input file describing
the fragmentation pattern. Use the provided methods to vibrate,
explode, and analyze the resulting data.

Dependencies
------------
- Python >= 3.7
- NumPy
- SciPy
- Pandas
- Mendeleev (for atomic weights)
- Matplotlib (optional, if you do plots here)

"""

import numpy as np
import re
import pandas as pd
from scipy.constants import physical_constants, c, hbar, epsilon_0
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from scipy.stats import truncnorm, norm
import itertools
from copy import deepcopy
import mendeleev as m

# Physical constants
U_TO_KG = physical_constants['atomic mass unit-kilogram relationship'][0]  # 1 amu in kg
E_CHARGE = 1.602176634e-19                                                 # Elementary charge (C)
K_COULOMB = E_CHARGE**2 / (4 * np.pi * epsilon_0)                          # e^2/(4πε0)

##############################################################################
# Helper Functions
##############################################################################

def symbol_to_weight(symbol: str) -> float:
    """
    Returns the atomic mass of the element with the given symbol (e.g. 'O', 'C').
    Uses the 'mendeleev' package internally.
    """
    symbol = symbol.strip()
    els = m.get_table('elements')
    ptable = els[['atomic_weight', 'symbol']]
    w = ptable.loc[ptable.symbol == symbol].atomic_weight
    return float(w)


def atom_number_to_mol_weight(atom_number: int) -> float:
    """
    Returns the atomic mass of the element with the given atomic number.
    Uses the 'mendeleev' package internally.
    """
    els = m.get_all_elements()
    return els[atom_number - 1].atomic_weight


def split_list_by_delimiter(lst, delimiter):
    """
    Generator that splits a list at the specified delimiter.

    Example:
    --------
    >>> test_list = ['A', 'B', 'OUTCOME', 'C', 'D', 'OUTCOME', 'E']
    >>> result = list(split_list_by_delimiter(test_list, 'OUTCOME'))
    >>> print(result)
    [['A', 'B'], ['C', 'D'], ['E']]
    """
    sublist = []
    for item in lst:
        if item == delimiter:
            yield sublist
            sublist = []
        else:
            sublist.append(item)
    yield sublist


def get_list_of_arrays(flat_list, n=3):
    """
    Splits a flat list of length 3N into a list of N arrays each of length 3.

    Example:
    --------
    >>> flat = [1, 2, 3, 4, 5, 6]
    >>> get_list_of_arrays(flat, n=3)
    [array([1, 2, 3]), array([4, 5, 6])]
    """
    return [np.array(flat_list[i:i+n]) for i in range(0, len(flat_list), n)]


##############################################################################
# I/O Functions (Reading Gaussian Files, etc.)
##############################################################################

def get_geom(filename: str) -> pd.DataFrame:
    """
    Reads the final optimized geometry from a Gaussian log file (Standard Orientation)
    and returns a DataFrame with columns: [a, x, y, z],
    where 'a' is the atomic mass and x,y,z are positions in Angstrom.

    Raises:
    -------
    Exception if the route section does not contain 'opt' or if not found.
    """
    # Read entire file
    with open(filename, 'r') as file:
        whole_file = file.read()

    # Check if route section has 'opt'
    route_match = re.search(r'#(.+?)\n', whole_file)
    if not route_match:
        raise ValueError('No route section found. Is this a Gaussian log file?')

    route_section = route_match.group(1)
    if 'opt' not in route_section.lower():
        raise ValueError('Geometry extraction requires an optimization job (no "opt" in route)')

    # We locate the lines after "Standard orientation:" and before the next line of '---'
    lines = whole_file.split('\n')
    start_idx, end_idx = None, None
    for i, line in enumerate(lines):
        if "Standard orientation:" in line:
            start_idx = i

    # After finding the last 'Standard orientation:', 
    # the geometry is 5 lines below it, until '---'
    if start_idx is None:
        raise ValueError("Couldn't find 'Standard orientation:' in log file.")

    for j in range(start_idx + 5, len(lines)):
        if '---' in lines[j]:
            end_idx = j
            break

    # Extract the geometry block
    geom_data = []
    for line in lines[start_idx + 5 : end_idx]:
        parts = line.split()
        # Format: [ (unused), atno, (unused), x, y, z ]
        # Gaussian standard orientation line format is often:
        #   Center  Atomic  Atomic             Coordinates (Angstroms)
        #   Number  Number   Type       X           Y           Z
        #    1       8       0        0.000000    0.000000    0.111111
        # parts[1] is atomic number, parts[3..5] are x,y,z
        at_mass = atom_number_to_mol_weight(int(parts[1]))
        x, y, z = map(float, parts[3:6])
        geom_data.append([at_mass, x, y, z])

    df = pd.DataFrame(geom_data, columns=['a', 'x', 'y', 'z'])
    return df


def extract_normal_modes(filename: str, n_atoms: int):
    """
    Extract normal modes from a Gaussian log file that has 'freq' 
    in the route section. It also requires that the file was run 
    with the 'P' option so that normal mode cartesian data 
    is printed in the log file.

    Returns:
    --------
    modes: dict[mode_number] = list of length N_atoms, each entry a 3-vector
    freqsdict: dict[mode_number] = float, frequency in cm^-1
    redmassesdict: dict[mode_number] = float, reduced mass in amu
    """
    if not filename.lower().endswith('.log'):
        raise ValueError("extract_normal_modes requires a Gaussian .log file")

    with open(filename, 'r') as file:
        whole_file = file.read()

    # Route extraction check
    route_match = re.search(r'#(.+?)\n', whole_file)
    if not route_match:
        raise ValueError('No route section found while extracting vibrational info.')

    route_section = route_match.group(1)
    if 'freq' not in route_section.lower():
        raise ValueError('Not a frequency calculation from Gaussian (no "freq" in route).')

    # The normal mode data is found between "and normal coordinates:" and " -------------------"
    try:
        normal_block = whole_file.split('and normal coordinates:')[1].split(' -------------------')[0]
    except IndexError:
        raise ValueError("Normal coordinate section not found; ensure the file was run with 'freq' and 'P' option.")

    rows = [line for line in normal_block.split('\n') if line.strip()]
    # Each block has length = 7 + n_atoms, repeated. Usually 3 modes per block.

    block_length = 7 + n_atoms
    # The data lines are typically chunked in sets of block_length
    # Each block corresponds to up to 3 normal modes.

    # Parse blocks
    modes = {}
    freqsdict = {}
    redmassesdict = {}

    # Break into sub-blocks of length block_length
    blocks = [rows[i : i + block_length] for i in range(0, len(rows), block_length)]

    for block in blocks:
        # 1) mode numbers
        mode_refs = block[0].split()
        # 2) frequencies (line 3 in Gaussian: block[2]) => typically 3 freq values
        freq_vals = block[2].split()[-len(mode_refs):]
        freq_vals = list(map(float, freq_vals))

        # 3) reduced masses (line 4 in Gaussian: block[3]) => typically 3 values
        redm_vals = block[3].split()[-len(mode_refs):]
        redm_vals = list(map(float, redm_vals))

        # Assign freq and reduced masses to each mode ref
        for i_mode, mode_number in enumerate(mode_refs):
            modes[mode_number] = [[] for _ in range(n_atoms)]
            freqsdict[mode_number] = freq_vals[i_mode]
            redmassesdict[mode_number] = redm_vals[i_mode]

        # Now parse the next lines containing the directional cosines
        # block lines 7.. => normal mode vectors for each atom
        coords_block = block[7:]
        # Each line has: atom_index, ???, x_mode1, y_mode1, z_mode1, x_mode2, ...
        for atom_line in coords_block:
            parts = atom_line.split()
            atom_idx = int(parts[0]) - 1  # Usually 1-based index in Gaussian
            # parts[2:] -> the displacement vectors for up to 3 modes
            # chunk them in 3
            # e.g. for 3 modes => we have 9 displacement values
            disp_vals = list(map(float, parts[2:]))

            # Each mode gets 3
            chunked_disp = [disp_vals[i : i + 3] for i in range(0, len(disp_vals), 3)]
            for i_mode, mode_number in enumerate(mode_refs):
                modes[mode_number][atom_idx] = chunked_disp[i_mode]

    return modes, freqsdict, redmassesdict


##############################################################################
# Classes
##############################################################################

class atom:
    """
    Represents a single atom with a position (pos) and a mass (mass).
    """
    def __init__(self, x, y, z, m):
        self.mass = float(m)  # in amu
        self.pos = np.array([x, y, z], dtype=float)
        self.frag_no = None   # which fragment it belongs to


class fragment:
    """
    Represents a fragment (ion) formed after Coulomb explosion,
    with multiple atoms or a single atom. Has a total mass, charge,
    and center-of-mass position.

    Attributes:
    -----------
    atoms : list[atom]
        Atoms belonging to this fragment.
    mass : float
        Total mass of the fragment in amu.
    q : int
        Charge of the fragment in +e units (e.g. +1, +2).
    pos : np.ndarray
        The 3D center-of-mass position (Angstroms).
    vx, vy, vz : floats
        Velocity components (m/s).
    vel : np.ndarray
        Full velocity vector [vx, vy, vz].
    force : np.ndarray
        Force vector [Fx, Fy, Fz] in SI units (Newtons).
    """
    def __init__(self, atoms_list):
        self.atoms = atoms_list
        self.mass = sum(atom.mass for atom in atoms_list)  # total mass in amu
        self.q = 0
        self.pos = None
        self.vel = np.zeros(3, dtype=float)
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
        self.force = np.zeros(3, dtype=float)
        self._center_of_mass()

    def _center_of_mass(self):
        """
        Calculate the center of mass from atoms (in Angstrom).
        """
        if len(self.atoms) == 1:
            self.pos = self.atoms[0].pos
        else:
            total_mass = 0.0
            weighted_pos_sum = np.zeros(3, dtype=float)
            for a in self.atoms:
                total_mass += a.mass
                weighted_pos_sum += a.mass * a.pos
            self.pos = weighted_pos_sum / total_mass


class molecule:
    """
    Holds the geometry (list of atoms) and associated vibrational data.
    Can be fragmented into ionic fragments (self.ions).

    Attributes:
    -----------
    info : pd.DataFrame
        The original DataFrame with columns ['a', 'x', 'y', 'z'].
    atoms : list[atom]
        The list of atom objects (internal representation).
    equilibrium_pos : np.ndarray
        The equilibrium positions of all atoms (Angstrom).
    modes : dict
        Dictionary of normal mode displacement vectors.
    freqsdict : dict
        Dictionary of frequencies in cm^-1.
    redmasses : dict
        Dictionary of reduced masses for each normal mode.
    ions : list[fragment]
        The list of fragment objects formed after explosion.
    """
    def __init__(self, geometry: pd.DataFrame):
        self.info = geometry.copy()
        self.masses = self.info['a'].values      # amu
        self.x_coords = self.info['x'].values    # Angstrom
        self.y_coords = self.info['y'].values    # Angstrom
        self.z_coords = self.info['z'].values    # Angstrom
        self.atoms = []
        self.equilibrium_pos = None

        # Vibrational data placeholders (populated externally)
        self.modes = {}
        self.freqsdict = {}
        self.redmasses = {}
        self.ions = []

        self._group_atoms()  # initialize self.atoms and equilibrium_pos

    def _group_atoms(self):
        """
        Internal method to create atom objects for each row in self.info
        and store them in self.atoms. Also keep a copy of eq positions.
        """
        self.atoms = []
        for i in range(len(self.info)):
            a = atom(self.x_coords[i], self.y_coords[i], self.z_coords[i], self.masses[i])
            self.atoms.append(a)

        # Keep track of equilibrium positions
        self.equilibrium_pos = np.array([atom.pos for atom in self.atoms])

    def vib(self):
        """
        Samples a quantum harmonic oscillator distribution for each normal mode
        and displaces the atoms accordingly. 
        Uses a 1D normal with sigma = (hbar / (2 * mu * omega))^(1/2).
        """
        for mode_number, direction_list in self.modes.items():
            freq = self.freqsdict[mode_number]        # in cm^-1
            redmass = self.redmasses[mode_number]     # in amu

            # Convert freq to rad/s: freq(cm^-1) * c(m/s) * 100
            omega = 2.0 * np.pi * freq * c * 100.0
            # Convert reduced mass to kg
            mu_kg = redmass * U_TO_KG

            # Quantum SHO standard deviation in position:
            #   σ = sqrt(ħ / (2 * mu * ω))
            sigma_pos = np.sqrt(hbar / (2.0 * mu_kg * omega)) * 1e10  # in Angstrom

            # Sample a random displacement from Gaussian(0, sigma_pos)
            displacement = norm(loc=0, scale=sigma_pos).rvs()

            # Displace each atom along that mode
            for i_atom, direction in enumerate(direction_list):
                self.equilibrium_pos[i_atom] += displacement * np.array(direction)

        # Update actual atom positions
        for i, atom_obj in enumerate(self.atoms):
            atom_obj.pos = self.equilibrium_pos[i]

    def create_fragments(self, channel_idx, expmnt):
        """
        Break the molecule into fragments according to the 
        fragment definitions in expmnt.channel_ions[channel_idx].
        Also assign charges from expmnt.channel_charges[channel_idx].

        The resulting fragments are stored in self.ions.
        """
        # Which atoms go to which fragment
        atoms_in_fragment = expmnt.channel_ions[channel_idx]
        # The charge on each fragment
        fragment_charges = expmnt.channel_charges[channel_idx]

        self.ions = []
        n_fragments = max(atoms_in_fragment.keys())

        # Build each fragment
        for frag_no in range(1, n_fragments + 1):
            idx_list = atoms_in_fragment[frag_no]  # the indices of atoms in this fragment
            frag_atoms = [self.atoms[i] for i in idx_list]
            frag = fragment(frag_atoms)
            frag.q = fragment_charges[frag_no]
            self.ions.append(frag)


class Experiment:
    """
    Main class that stores:
    - The geometry and normal modes from Gaussian
    - The fragmentation channels and probabilities
    - The molecule object with eq geometry
    - Tools for generating random event rates (Poisson + truncated normal)

    Attributes:
    -----------
    compchemfile : str
        Gaussian filename for geometry and vibrational info.
    geometry : pd.DataFrame
        DataFrame of the final geometry with columns [a, x, y, z].
    molecule_object : molecule
        A base molecule object with eq geometry (used as a template).
    numatoms : int
        Number of atoms in the system.
    modes, freqsdict, redmassesdict : dict
        Data for each normal mode, frequency, and reduced mass.

    channel_ions : dict of dict
        channel_ions[i_channel][fragment_number] => list of atom indices.
    channel_probs : list[float]
        Probability of each channel. Must sum to 1.
    channel_charges : dict of dict
        channel_charges[i_channel][fragment_number] => integer charge of fragment.

    sigma : float
        The sigma used for truncated normal distribution of event rate.
    v0 : float
        The default or nominal event rate for the Poisson distribution.

    Methods:
    --------
    assign_charges(channels):
        Used internally by read_instructions to set up the channel data.
    create_molecule_obj():
        Returns a deep copy of self.molecule_object for further manipulations.
    normaldist(sigma, upper, lower, mu):
        Creates a truncated normal distribution with the given parameters.
    """

    def __init__(self, filename: str):
        self.compchemfile = filename

        # Read geometry and set up base molecule
        self.geometry = get_geom(filename)
        self.molecule_object = molecule(self.geometry)
        self.numatoms = len(self.molecule_object.atoms)

        # Extract normal mode info
        self.modes, self.freqsdict, self.redmassesdict = extract_normal_modes(filename, self.numatoms)

        # Link them into the base molecule
        self.molecule_object.modes = deepcopy(self.modes)
        self.molecule_object.freqsdict = deepcopy(self.freqsdict)
        self.molecule_object.redmasses = deepcopy(self.redmassesdict)

        # Placeholder for channel info
        self.no_channels = 0
        self.channel_ions = {}
        self.channel_probs = []
        self.channel_charges = {}

        # Default attributes
        self.name = None
        self.sigma = 0.0
        self.v0 = 1.0
        self.Gamma_dist = None

    def __repr__(self):
        return f'Experiment("{self.compchemfile}") object with {self.no_channels} channels.'

    def assign_charges(self, channels):
        """
        For each channel in 'channels', parse the fragments.
        channels is a list of lists with the structure:
        [
          [num_fragments, prob, <frag_assignment...>, <charges...>],
          [...],
          ...
        ]
        The frag_assignment is the fragment number for each atom in order.
        """
        self.channel_ions = {}
        self.channel_probs = []
        self.channel_charges = {}
        n_atoms = self.numatoms

        for i, ch in enumerate(channels):
            # ch[0] = no_fragments, ch[1] = probability
            no_fragments = int(ch[0])
            prob = float(ch[1])
            self.channel_probs.append(prob)

            # The next n_atoms entries specify each atom's fragment number
            frag_assignment = ch[2 : 2 + n_atoms]
            frag_assignment = list(map(int, frag_assignment))

            # The last no_fragments entries are the charges
            charges = ch[2 + n_atoms : 2 + n_atoms + no_fragments]
            charges = list(map(int, charges))

            # Build dictionary of fragment->list_of_atom_indices
            fragment_atoms = {frag_no: [] for frag_no in range(1, no_fragments + 1)}
            for atom_idx, fno in enumerate(frag_assignment):
                fragment_atoms[fno].append(atom_idx)

            # Build dictionary of fragment->charge
            fragment_charges = {frag_no: charges[frag_no - 1] for frag_no in range(1, no_fragments + 1)}

            self.channel_ions[i] = fragment_atoms
            self.channel_charges[i] = fragment_charges

        # Ensure probabilities sum to 1 (if needed you can normalize)
        total_prob = sum(self.channel_probs)
        if not np.isclose(total_prob, 1.0, atol=1e-6):
            # Optional: auto-normalize or raise an error
            pass

    def create_molecule_obj(self) -> molecule:
        """
        Returns a deep copy of the template molecule object.
        Preserves equilibrium geometry in the parent object.

        The returned object can safely be vibrated, rotated, fragmented, etc.
        """
        mol_copy = deepcopy(self.molecule_object)
        return mol_copy

    def normaldist(self, sigma, upper, lower, mu):
        """
        Initializes a truncated normal distribution with given bounds,
        center mu, and standard deviation sigma.
        """
        if sigma <= 0:
            self.Gamma_dist = None
            return None
        else:
            Gamma_dist = truncnorm(
                (lower - mu) / sigma,
                (upper - mu) / sigma,
                loc=mu,
                scale=sigma
            )
            self.Gamma_dist = Gamma_dist
            return Gamma_dist


##############################################################################
# Public API: ODE and Explosion Routines
##############################################################################

def forcecalc_ode(current_state, particles):
    """
    Computes forces among a set of charged fragments (Coulomb repulsion).
    current_state: array of shape (2*3N,) with positions followed by velocities.
                   positions are [x1, y1, z1, ..., xN, yN, zN]
                   velocities are appended.
    particles: list of fragment objects with attributes mass (amu) and q (integer charge).
               The function updates each fragment's .pos and .force.

    Returns the updated list of particles (with .force).
    """
    n = len(particles)

    # Split the current_state into positions and velocities
    pos, vel = np.split(current_state, 2)
    pos_vectors = get_list_of_arrays(pos, n=3)

    # Reset forces
    for p in particles:
        p.force[:] = 0.0

    # Update positions in the fragment objects
    for i, p in enumerate(particles):
        p.pos = pos_vectors[i]

    # Coulomb force pairwise
    for i in range(n):
        for j in range(i + 1, n):
            qi, qj = particles[i].q, particles[j].q
            if qi == 0 or qj == 0:
                continue
            ri = particles[i].pos
            rj = particles[j].pos
            diff = ri - rj
            dist = np.linalg.norm(diff)
            if dist < 1e-20:
                # Avoid singularities if two fragments overlap
                continue

            # Force magnitude = K_COULOMB * (qi*qj) / dist^2
            # Force vector = magnitude * (diff / dist)
            force_scalar = (K_COULOMB * qi * qj) / (dist**3)
            fij = force_scalar * diff

            # Add to each particle
            particles[i].force += fij
            particles[j].force -= fij

    return particles


def equations(t, current_state, *particles):
    """
    ODE system derivative function for solve_ivp.
    [dpos/dt, dvel/dt] = [vel, force/mass].
    """
    n = len(particles)
    # Compute forces at these positions
    updated_particles = forcecalc_ode(current_state, particles)

    # Extract accelerations
    acc = []
    for p in updated_particles:
        # mass in amu => convert to kg
        mass_kg = p.mass * U_TO_KG
        a = p.force / mass_kg
        acc.extend(a)
    acc = np.array(acc)

    pos, vel = np.split(current_state, 2)

    # Return concatenation of velocities and accelerations
    return np.concatenate([vel, acc])


def solve_equations(particles, t_eval=np.array([1e-8]), t_span=(0.0, 1e-8)):
    """
    Integrates the Coulomb explosion using solve_ivp with
    the 'equations' function above.

    particles: list of fragment objects (with pos, mass, q).
    t_eval: array of times at which to store the solution.
    t_span: (t0, tf) specifying start and final integration times.

    Returns:
    --------
    sol : OdeResult
        The integration solution object from solve_ivp.
    """
    n = len(particles)
    # Build initial_values [pos...pos, vel...vel]
    init_vals = []
    for p in particles:
        init_vals.extend(p.pos)  # x, y, z
    for p in particles:
        init_vals.extend([p.vx, p.vy, p.vz])  # vx, vy, vz

    init_vals = np.array(init_vals)
    sol = solve_ivp(
        fun=equations,
        t_span=t_span,
        y0=init_vals,
        t_eval=t_eval,
        args=particles,
        vectorized=False
    )
    if not sol.success:
        raise RuntimeError(f"Solve_ivp failed: {sol.message}")

    return sol


def calc_time_of_flight(q, mass, flight_tube_length=0.6, voltage=7000.0):
    """
    Approximate time-of-flight for a singly or multiply charged ion in a
    simple linear TOF. Not necessarily accurate for Velocity Map Imaging,
    but used as an example.

    t = L * sqrt(m / (2*q*e*V)), 
    where:
      L = flight tube length in meters,
      m = mass in kg,
      q = charge (integer * elementary charge),
      V = voltage in Volts.

    Returns time in seconds.
    """
    if q == 0:
        return 0.0
    mass_kg = mass * U_TO_KG
    return flight_tube_length * np.sqrt(mass_kg / (2.0 * q * E_CHARGE * voltage))


def rotate_fragments_randomly(fragments, seed=None):
    """
    Applies the same random 3D rotation to all fragments (i.e., 
    the molecule is randomly oriented in space).
    """
    rotator = Rotation.random(random_state=seed)
    for frg in fragments:
        frg.pos = rotator.apply(frg.pos)
    return fragments


def init_vel_to_zero(fragments):
    """
    Initialize all fragment velocities to zero (vx=vy=vz=0).
    """
    for frg in fragments:
        frg.vx, frg.vy, frg.vz = 0.0, 0.0, 0.0
        frg.vel = np.zeros(3, dtype=float)
    return fragments


def explode_fragments(molecules, epsilon=1.0, tf=1e-8):
    """
    Performs the Coulomb explosion for each molecule in 'molecules'
    by:
      1) Converting positions from Angstrom to meters.
      2) Rotating the entire molecule randomly.
      3) Integrating to time tf.
      4) Computing final time-of-flight or approximate detection.

    Only ions with q != 0 have forces and are included in the integration.
    A random detection filter with probability 'epsilon' is applied:
    If a random number is > epsilon, the ion is not "detected".

    Returns:
    --------
    (xs, ys, zs, ts, ms, qs, vxs, vys, vzs)
    Each is a Python list.
    """
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng()

    xs, ys, zs = [], [], []
    ts, ms, qs = [], [], []
    vxs, vys, vzs = [], [], []

    for mol in molecules:
        # Each molecule has a list: mol.ions
        # Filter out uncharged (q=0) - no force, no flight time
        ions = [ion for ion in mol.ions if ion.q != 0]
        if not ions:
            continue

        # Initialize velocities
        ions = init_vel_to_zero(ions)

        # Random orientation
        seed_val = rng.integers(1e6)  # or None
        ions = rotate_fragments_randomly(ions, seed=seed_val)

        # Convert pos from Angstrom to meter
        for frg in ions:
            frg.pos *= 1e-10

        # Solve ODE to time tf
        sol = solve_equations(ions, t_eval=np.array([tf]), t_span=(0.0, tf))
        final_positions, final_velocities = np.split(sol.y, 2)

        # final_positions has shape (3N, 1), final_velocities has shape (3N, 1)
        # reorganize into [ [x1,y1,z1], [x2,y2,z2], ... ]
        final_pos_arrays = get_list_of_arrays(final_positions[:, 0], n=3)
        final_vel_arrays = get_list_of_arrays(final_velocities[:, 0], n=3)

        # Assign back
        for i, frg in enumerate(ions):
            frg.pos = final_pos_arrays[i]
            frg.vel = final_vel_arrays[i]
            frg.vx, frg.vy, frg.vz = frg.vel

        # Now approximate time-of-flight from t=tf to the detector
        for frg in ions:
            tof = calc_time_of_flight(frg.q, frg.mass)
            # If you want to push the fragment from tf -> tof using its velocity:
            # final_pos = frg.pos + frg.vel*(tof - tf)
            # For a simpler approach, just store the position at tf or do nothing.

            # Detection filter
            if rng.random() > epsilon:
                continue

            xs.append(frg.pos[0])
            ys.append(frg.pos[1])
            zs.append(frg.pos[2])
            ts.append(tof)
            ms.append(frg.mass)
            qs.append(frg.q)
            vxs.append(frg.vx)
            vys.append(frg.vy)
            vzs.append(frg.vz)

    return xs, ys, zs, ts, ms, qs, vxs, vys, vzs


##############################################################################
# Public Function: read_instructions
##############################################################################

def read_instructions(file_path: str) -> Experiment:
    """
    Reads a text file that specifies simulation parameters, e.g.:

    (1)   Name of experiment
    (2)   Gaussian log file name
    (3)   number_of_channels
    (4)   v0 (average event rate)

    Then multiple 'OUTCOME' lines specifying each fragmentation channel:
    Each channel line:
        no_fragments, probability, <frag_assignment_for_each_atom>, <charges_for_each_fragment>

    The last line must be 'END'.

    Returns:
    --------
    species : Experiment
        An Experiment object storing geometry, channels, etc.
    """
    with open(file_path, 'r') as f:
        raw_lines = f.readlines()

    # Remove comments (#) and strip whitespace
    instructions = []
    for line in raw_lines:
        line = line.split('#', 1)[0].strip()
        if line:
            instructions.append(line)

    # 0: name, 1: compchemfile, 2: no_channels, 3: v0
    name = instructions[0]
    compchemfile = instructions[1]
    no_channels = int(instructions[2])
    v0 = float(instructions[3])

    if instructions[-1].upper() != 'END':
        raise ValueError("Instruction file must end with 'END'")

    # Create the Experiment
    species = Experiment(compchemfile)
    species.no_channels = no_channels
    species.name = name
    species.v0 = v0

    # The channel definitions appear after the 4 lines above.
    # They are separated by the keyword 'OUTCOME'.
    # Let's use our helper:
    splitted = list(split_list_by_delimiter(instructions, 'OUTCOME'))
    # splitted[0] => the lines up to the first OUTCOME (already used: name, file, channels, v0)
    # splitted[1..] => each channel's specification
    channels_data = splitted[1:]  # ignoring the first sublist

    # Each sublist in channels_data is one channel's info
    # Example format:
    # [ str(no_fragments), str(prob), N_atoms_of_fragment_assignments..., fragment_charges...,  (maybe more lines)... ]
    # We'll flatten them carefully.
    channels = []
    for ch_sublist in channels_data:
        # ch_sublist is typically a single line unless there's formatting differences
        # e.g.: [ '2', '0.5', '1', '1', '2', '2', '1', '1' ]
        # Flatten if needed:
        flat = []
        for c in ch_sublist:
            flat.extend(c.split())
        channels.append(flat)

    # Now assign them
    species.assign_charges(channels)

    return species
