import numpy as np
from ase.io import read,write
import re
import dpdata
import os


def get_calc_time(filepath:str):
    # Get calculation time from quantum espresso output file
    f = open(filepath,'r')
    try:
        pwo_lines = f.readlines()
    except:
        raise RuntimeError("Please specify the correct file")
    time_line = pwo_lines[-8].split()
    hour_list = [i for i in time_line if ('h' in i and not('m' in i) and not('s' in i))]
    min_list = [i for i in time_line if ('m' in i and not('h' in i) and not('s' in i))]
    sec_list = [i for i in time_line if ('s' in i and not('h' in i) and not('m' in i))]
    min_sec_list = [i for i in time_line if (('m' in i) and ('s' in i) and not('h' in i))]
    hour_min_list = [i for i in time_line if (('h' in i) and ('m' in i) and not('s' in i))]
    hour_min_sec_list = [i for i in time_line if (('h' in i) and ('m' in i) and ('s' in i))]

    if len(hour_list)==0:
        hour_list.append('0h')
    if len(min_list)==0:
        min_list.append('0m')
    if len(sec_list)==0:
        sec_list.append('0s')
    if len(hour_min_list)==0:
        hour_min_list.append('0h0m')
    if len(min_sec_list)==0:
        min_sec_list.append('0m0s')
    if len(hour_min_sec_list)==0:
        hour_min_sec_list.append('0h0m0s')
    hour = np.array(re.findall(r"(\d+)h",hour_list[0])+re.findall(r"(\d+)h",hour_min_list[-1])+re.findall(r"(\d+)h",hour_min_sec_list[-1]),dtype=float)
    min = np.array(re.findall(r"(\d+)m",min_list[-1])+re.findall(r"(\d+)m",hour_min_list[-1])+re.findall(r"(\d+)m",min_sec_list[-1])+re.findall(r"(\d+)m",hour_min_sec_list[-1]),dtype=float)
    second = np.array(re.findall(r"(\d*\.\d*)s",sec_list[-1])+re.findall(r"(\d*\.\d*)s",min_sec_list[-1])+re.findall(r"(\d*\.\d*)s",hour_min_sec_list[-1]),dtype=float)
    time = np.sum(hour)*3600+np.sum(min)*60+np.sum(second)
    return time

def force_distribution(atoms_list,expand=True,**kwargs):
    """
    Get force distribution of training dataset

    Parameters
    ----------
        atoms_list: list or numpy.ndarray
                    list storing atoms to be fed into training
        expand: bool 
                    whether to expand the force distribution array

    Returns 
    ----------
        force_list: numpy.ndarray
                    numpy array containing force distribution
    """
    from ase.atoms import Atoms
    if not isinstance(atoms_list[0],Atoms):
        raise RuntimeError("Only support atom list read in!")
    force_list = np.array([atoms.get_forces(**kwargs) for atoms in atoms_list])
    if expand==True:
        force_list = force_list.reshape(len(atoms_list)*len(atoms_list[0]*3))
    return force_list
    
def energy_distribution(atoms_list,**kwargs):
    """
    Get energy distribution of training dataset

    Parameters
    ----------
        atoms_list: list or numpy.ndarray
                    list storing atoms to be fed into training

    Returns 
    ----------
        energy_list: numpy.ndarray
                    numpy array containing energy distribution
    """
    from ase.atoms import Atoms
    if not isinstance(atoms_list[0],Atoms):
        raise RuntimeError("Only support atom list read in!")
    energy_list = np.array([atoms.get_potential_energy(**kwargs) for atoms in atoms_list])
    return energy_list

def energy_per_fus(atoms_list, fus:str, **kwargs):
    """
    Get energy per formulation unit (fus) of training dataset

    Parameters
    ----------
        atoms_list: list or numpy.ndarray
            list storing atoms to be fed into training
        fus: str
            formation unit, element symbol+number (e.g. C1F1, Li1)

    Returns 
    ----------
    energy_per_fus_list: numpy.ndarray
            numpy array containing energy per fus distribution
    """
    from ase.atoms import Atoms
    if not isinstance(atoms_list[0],Atoms):
        raise RuntimeError("Only support atom list read in!")
    # get the total number of atoms in fus
    number_list = re.findall(r'[a-zA-Z]+(\d)',fus)
    number_fus = len(atoms_list[0])/np.sum(np.array([int(n) for n in number_list]))
    energy_per_fus_list = np.array([atoms.get_potential_energy(**kwargs)/number_fus for atoms in atoms_list])
    return energy_per_fus_list

def stress_distribution(atoms_list,voigt=True,**kwargs):
    """
    Get energy distribution of training dataset

    Parameters
    ----------
        atoms_list: list or numpy.ndarray 
                    list storing atoms to be fed into training
        voigt: bool, default=True
                    stress tensor in Voigt order (xx,yy,zz,yz,xz,xy) or not

    Returns
    ----------
        force_list: numpy.ndarray
        numpy array containing distribution
    """
    from ase.atoms import Atoms
    if not isinstance(atoms_list[0],Atoms):
        raise RuntimeError("Only support atom list read in!")
    energy_list = np.array([atoms.get_potential_stress(voigt=voigt,**kwargs) for atoms in atoms_list])
    return energy_list

def InitSystemGen(system:dpdata.system.System,perturb_list):
    """
    Generate perturbed system for constructing training set

    Parameters
    ----------
        system: dpdata.system.System 
                baseline system for perturbation
        perturb_list: list or numpy.ndarray
                perturb parameters, 
                list/array format: [number, cell perturb, atom perturb, atom perturb style]

    Returns
    ----------
        system: dpdata.system.System
                perturbed system that can be further fed to dft calculation and generate training set
    """
    if not(isinstance(system, dpdata.system.System)):
        raise RuntimeError("Need to specify initial system first!")
    
    system_copy = system.copy()

    for i in range(len(perturb_list)):
        pert_num = perturb_list[i][0]
        cell_pert_fraction = perturb_list[i][1]
        atom_pert_distance = perturb_list[i][2]
        atom_pert_style = perturb_list[i][3]

        perturbed_system = system.perturb(
            pert_num = pert_num,
            cell_pert_fraction= cell_pert_fraction,
            atom_pert_distance= atom_pert_distance,
            atom_pert_style= atom_pert_style
        )

        system_copy = system_copy + perturbed_system

    return system_copy

def InitModelDevi(train_system_path:str, num_model_devi_init:int, output_dir:str, output_format='vasp/poscar'):
    system = dpdata.LabeledSystem(train_system_path, 'deepmd/raw')
    random_list = np.random.randint(len(system),num_model_devi_init)

    if ('poscar' in output_format):
        for i in range(len(random_list)):
            if os.path.exists(output_dir+f'/{i+1:06d}')==False:
                os.makedirs(output_dir+f'/{i+1:06d}')
            system[int(random_list[i])].to(output_dir+f"/{i+1:06d}/POSCAR")
    else:
        print('Other formats have not been implemented yet')
        pass

def get_kpoints(system:dpdata.system.System,kspacing:float,atoms_list=None):
    if (system is None) and (atoms_list is None):
        raise RuntimeError("Please specify system!")

    if (isinstance(system,dpdata.system.System)):
        atoms_list = system.to("ase/structure")
        
    cell_list = [atoms.get_cell() for atoms in atoms_list]
    rcell_list = [np.linalg.inv(np.array(cell)) for cell in cell_list]
    rcell_list = [rcell.T for rcell in rcell_list]
    kpoints = [[np.ceil(2*np.pi*np.linalg.norm(ii)/kspacing).astype(int) for ii in rcell] for rcell in rcell_list]

    return kpoints

def TrainTestSplit():
    pass

