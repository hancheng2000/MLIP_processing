from MLIP_preprocessing import utils
import numpy as np
import os
from ase.io import read,write
from ase.io.espresso import write_espresso_in
import dpdata


def InitJobGen(
    system: dpdata.system.System, 
    pseudopotentials: dict,
    pseudo_dir:str,
    cutoff_energy:int,
    input_data=None,
    atoms_list=None,
    mode = 'scf',
    kspacing = 0.18,
    kpoints = None,
    to_dir = None,
    starting_index=1,
    ):
    """
    Generate init jobs to submit to HPC for labeling

    Parameters
    ----------
    system: dpdata.System.system
            init system prepared by either user or InitSystemGen
    atoms_list: list or numpy.ndarray
            list containing atoms for init jobs, will be ignored if system specified
    pseudopotentials: dict
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
    to_dir: str
        parent directory to write input file to

    Returns
    -------
    None
    """
    if not(isinstance(system,dpdata.system.System)) and (atoms_list is None):
        raise RuntimeError("No system is specified!")
    if isinstance(system,dpdata.system.System):
        atoms_list = system.to('ase/structure')

    if mode=='scf' and (input_data is None):
        input_data = {
                'calculation': 'scf',
                'nstep': 1,
                'pseudo_dir': pseudo_dir,
                'ibrav':0,
                'ecutwfc': cutoff_energy,
                'conv_thr': 1.0e-6,
                'forc_conv_thr': 1.0e-3,
                'etot_conv_thr': 1.0e-4,
                'disk_io':'nowf',
                'tprnfor':True,
                'tstress':True,
                'electron_maxstep':200,
        }
    elif mode=='vc-relax' and (input_data is None):
        input_data = {
                'calculation': 'vc-relax',
                'nstep': 200,
                'pseudo_dir': pseudo_dir,
                'ibrav':0,
                'ecutwfc': cutoff_energy,
                'conv_thr': 1.0e-6,
                'forc_conv_thr': 1.0e-3,
                'etot_conv_thr': 1.0e-4,
                'press_conv_thr': 0.5,
                'disk_io':'nowf',
                'tprnfor':True,
                'tstress':True,
                'electron_maxstep':200,
                'ion_dynamics':'bfgs',
                'cell_dynamics':'bfgs',
        }
    elif input_data==None:
        raise RuntimeError("Please specify input data")

    kpoints_list = utils.get_kpoints(system, kspacing = kspacing)

    for i in range(0,len(system)):
        dir = to_dir+f"/{i+starting_index}/pbe"
        if os.path.exists(dir)==False:
            os.makedirs(dir)

        input_data['nat'] = len(atoms_list[i])
        input_data['ntyp'] = len(atoms_list[i].get_atomic_numbers())
        write_espresso_in(open(dir+'/input','w'), atoms = atoms_list[i],
                          input_data = input_data,
                          pseudopotentials = pseudopotentials,
                          kpts = kpoints_list[i]
                          )

    return None

    