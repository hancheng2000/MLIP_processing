from MLIP_preprocessing import utils
import numpy as np
import os
from ase.io import read,write
from ase.io.espresso import write_espresso_in
import dpdata


def InitJobGen(
    system: dpdata.System.system, 
    atoms_list,
    pseudopotential: dict,
    cutoff_energy: int,
    mode = 'scf',
    input_data: dict
    kspacing = 0.18,
    kpoints = None,
    ):
    """
    Generate init jobs to submit to HPC for labelling

    Parameters
    ----------
    system: dpdata.System.system
            init system prepared by either user or InitSystemGen
    atoms_list: list or numpy.ndarray
            list containing atoms for init jobs, will be ignored if system specified
    pseudopotential: dict
            dictionary containing pseudopotential file
    cutoff_energy: int
            cutoff energy for PW calculation (unit: Ry)
    mode: str (optional)
            calculation mode, default = scf
    input_data: dict
            dictionary containing data written into input
    kspacing: float
            specify kspacing in k-space, if kpoints defined then ignored
    kpoints: list
            specify kpoints, will ignore kspacing if this parameter defined
     

    Returns
    -------
    None
    """
    if not(istype(system,dpdata.system.System)) and (atoms_list is None):
        raise RuntimeError("No system is specified!")
    
