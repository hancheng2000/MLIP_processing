import numpy as np
from ase.io import read,write
import re
import dpdata
import os
from ase import db
import pandas as pd
from ase.db import connect
import matplotlib.pyplot as plt
from ase.eos import EquationOfState
from ase.units import kJ

def get_calc_time(filepath:str):
    # Get calculation time from quantum espresso output file
    with open(filepath,'r') as f:
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
    force_list = [atoms.get_forces(**kwargs) for atoms in atoms_list]
    if expand==True:
        force_list = np.hstack([atoms.get_forces(**kwargs).flatten() for atoms in atoms_list])
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
    # number_fus = len(atoms_list[0])/np.sum(np.array([int(n) for n in number_list]))
    number_fus = np.sum(np.array([int(n) for n in number_list]))
    print(number_fus)
    energy_per_fus_list = np.array([atoms.get_potential_energy(**kwargs)/len(atoms)*number_fus for atoms in atoms_list])
    return energy_per_fus_list

def stress_distribution(atoms_list,voigt=True,expand=True,**kwargs):
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
    stress_list = np.array([atoms.get_stress(voigt=voigt,**kwargs) for atoms in atoms_list])
    if expand:
        stress_list = stress_list.flatten()
    return stress_list

def volume_distribution(atoms_list,**kwargs):
    """
    Get volume distribution of training dataset

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
    volume_list = np.array([atoms.get_volume(**kwargs) for atoms in atoms_list])
    return volume_list

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
    random_list = np.random.randint(len(system), size = num_model_devi_init)

    if ('poscar' in output_format):
        for i in range(len(random_list)):
            if not os.path.exists(output_dir+f'/{i:06d}'):
                os.makedirs(output_dir+f'/{i:06d}')
            system[int(random_list[i])].to('vasp/poscar',output_dir+f"/{i:06d}/POSCAR")
    else:
        print('Other formats have not been implemented yet')
        pass

def get_kpoints(kspacing:float,atoms_list=None,system:dpdata.system.System=None, even=True):
    if (system is None) and (atoms_list is None):
        raise RuntimeError("Please specify system!")

    if (isinstance(system,dpdata.system.System)):
        atoms_list = system.to("ase/structure")
        
    cell_list = [atoms.get_cell() for atoms in atoms_list]
    rcell_list = [np.linalg.inv(np.array(cell)) for cell in cell_list]
    rcell_list = [rcell.T for rcell in rcell_list]
    kpoints = [[np.ceil(2*np.pi*np.linalg.norm(ii)/kspacing).astype(int) for ii in rcell] for rcell in rcell_list]
    if even:
        kpoints = np.array(kpoints).flatten()
        kpoints = np.array([kpts+1 if kpts%2!=0 else kpts for kpts in kpoints]).reshape((len(rcell_list),3))

    return kpoints

def TrainTestValSplit(system,train_ratio=0.8,test_ratio=0.2,val_ratio=0.0,shuffle=False,train_index=None,test_index=None):
    if train_ratio+test_ratio+val_ratio!=1.0:
        val_ratio = 1.0 - train_ratio - test_ratio
    assert train_ratio+test_ratio+val_ratio==1.0
    train_size = np.floor(len(system)*train_ratio).astype(int)
    test_size = np.floor(len(system)*test_ratio).astype(int)
    val_size = (len(system)-train_size-test_size).astype(int)
    # if not(train_index==None):
    #     train_system = system[train_index]
    #     test_system = system - train_system
    if shuffle==False:
        train_system = system[:train_size]
        test_system = system[train_size:train_size+test_size]
        val_system = system[train_size+test_size:]
    else:
        if type(system)==dpdata.system.System or type(system)==dpdata.system.LabeledSystem or type(system)==dpdata.system.MultiSystems:
            shuffle_list = system.shuffle().tolist()
            system_shuffle = system[shuffle_list]
            train_system = system[:train_size]
            test_system = system[train_size:train_size+test_size]
            val_system = system[train_size+test_size:]
        elif type(system)==list:
            import random
            random.shuffle(system)
            train_system = system[:train_size]
            test_system = system[train_size:train_size+test_size]
            val_system = system[train_size+test_size:]
            print(len(train_system),len(test_system),len(val_system))
    return train_system, test_system, val_system

def TrainSystemGen(database:str,format:str,dstpath:str,shuffled=False):
    from tqdm import tqdm
    db_temp = db.connect(database)
    system = dpdata.LabeledSystem()
    for i in tqdm(range(len(db_temp))):
        system = system + dpdata.LabeledSystem(read(database+f'@{i}'),'ase/structure')
    system_copy = system.copy()
    if shuffled==True:
        shuffle_list = system.shuffle().tolist()
        system_shuffle = system[shuffle_list]
        system_copy = system_shuffle.copy()
    system_copy.to(format,dstpath)

def plot_loss_curve(path, names=False, step=100, savefig=False, figpath=None, show=True):
    data = np.genfromtxt(path, names=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    data = np.array([data[i] for i in range(len(data)) if data[i]['step']%step==0])
    fig, ax = plt.subplots()
    if not names:
        for name in data.dtype.names[1:-1]:
            plt.plot(data['step'], data[name], label=name)
    else:
        for name in names:
            plt.plot(data['step'], data[name], label=name)
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Loss')
    # plt.xscale('log')
    plt.yscale('log')
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    plt.grid()
    if savefig==True:
        plt.savefig(figpath+"/loss.png",bbox_inches='tight')
    if show:
        plt.show()

def get_complete_dataset(init_path,run_path,iter_range=None,to_dir=None):
    system = dpdata.LabeledSystem(init_path,'deepmd/raw')
    if iter_range==None:
        iter = len(next(os.walk(run_path))[1])
        iter_range = range(iter-1)
    elif not isinstance(iter_range,range):
        iter_range = range(iter_range[0],iter_range[1])
    # Most common the last iteration only has 00.train so we ignore the last iteration by default
    for i in iter_range:
        iter_path = run_path + f"/iter.{i:06d}/02.fp"
        for root, dirs, files in os.walk(iter_path):
            if("data.0" in root and "set" not in root):
                system_temp = dpdata.LabeledSystem(root,'deepmd/raw')
                system = system + system_temp

    if to_dir:
        if not os.path.exists(to_dir):
            os.makedirs(to_dir,exist_ok=True)
        system.to('deepmd/raw',to_dir)
    return system

def plot_stats(db):
    fus = ['PH2', 'PH3', 'PH4', 'PH5', 'PH6']
    nfigs = 4
    props = {}
    bad = []
    for f, fu in enumerate(fus):
        props[fu] = {'en_per_fus': [], 'max_fs': [], 'densities': [], 'max_stresses': []}
        for i in range(nfigs):
            for row in db.select(fu=fu):
                num_fu = row.num_fu
                atoms = row.toatoms()
                props[fu]['en_per_fus'].append(row.energy / num_fu)
                props[fu]['max_fs'].append(np.max(atoms.get_forces()))
                props[fu]['max_stresses'].append(np.max(atoms.get_stress()))
                mass = np.sum(atoms.get_masses())
                vol = atoms.get_volume()
                props[fu]['densities'].append(mass / vol)

                if row.energy / num_fu > 40 or np.max(atoms.get_forces()) > 100:
                    bad.append(row.id)

    prop_df = pd.DataFrame.from_dict(props)

    for p, prop in enumerate(prop_df['PH2'].keys()):
        # print(prop)
        fig, axs = plt.subplots(nrows=len(fus), figsize=[5,12])
        for f, fu in enumerate(fus):
            ax = axs[f]
            ax.hist(prop_df[fu][prop], log=True, bins=100)
            ax.set_title(fu)
            ax.set_ylabel(prop)
        plt.savefig(f'{prop}.svg')
    return prop_df, bad

def qe_eos(path,num_dft=5, start = 2,mode=None, lattice = 'a', plot=True, plot_path=None,fus=None,**kwargs):
    num = num_dft
    scaleQE = np.zeros(num)
    energy_qe = np.zeros(len(scaleQE))
    volume_qe = np.zeros(len(scaleQE))
    for i in range(len(scaleQE)):
        try:
            atoms = read(path+f"/{i+start}/pbe/output")
            energy_qe[i] = atoms.get_potential_energy()
            volume_qe[i] = atoms.get_volume()
        except:
            print(f"{i+start} not finished")
    if not mode:
        mode = 'sjeos'
    if fus:
        number_list = re.findall(r'[a-zA-Z]+(\d)',fus)
        # number_fus = len(atoms_list[0])/np.sum(np.array([int(n) for n in number_list]))
        number_fus = np.sum(np.array([int(n) for n in number_list]))
        energy_qe = energy_qe / len(atoms) * number_fus

    if kwargs.get('relative',None):
        energy_qe = energy_qe - np.min(energy_qe)    
    try:
        eos_qe = EquationOfState(volume_qe,energy_qe,eos = mode)
        v0qe,e0qe,Bqe = eos_qe.fit()
        Bqe=Bqe/kJ*1.0e24
        cell = atoms.get_cell()
        if lattice == 'a':
            if np.abs(cell[0][0]-np.linalg.norm(cell[1]))>(np.linalg.norm(cell[1]/2)):
                a = np.sqrt(v0qe/np.sqrt(3)/cell[-1][-1])
            else:
                a = np.sqrt(2*v0qe/np.sqrt(3)/cell[-1][-1])
        elif lattice == 'c':
            if np.abs(cell[0][0]-np.linalg.norm(cell[1]))>(np.linalg.norm(cell[1]/2)):
                a = 4*v0qe/np.sqrt(3)/cell[0][0]**2
            else:
                a = 2*v0qe/np.sqrt(3)/cell[0][0]**2
        # print(f"{lattice}={a},v0={v0qe}")
        if plot:
            if kwargs['fig'] and kwargs['ax']:
                fig, ax = kwargs['fig'], kwargs['ax']
            else:
                fig, ax = plt.subplots()
            if kwargs['txt'] == True:
                plt.text(0.4,0.6,f"DFT\n"+r"$V_0$"+f"={v0qe:.4f}"+r"$\AA^3$"+f"\n{lattice}={a:.4f}Å\n",transform=ax.transAxes,wrap=True)
            # if not os.path.exists(plot_path):
            #     os.makedirs(plot_path)
            # print(os.getcwd(),plot_path)
            eos_qe.plot(plot_path,color=kwargs.get('color','blue'),label='DFT',mec=kwargs.get('mec','b'),mfc=kwargs.get('mfc','none'), ls=kwargs.get('ls','--'))
            if kwargs.get('close',True):
                plt.close(fig=fig)
    except ValueError: 
        if plot:
            if kwargs['fig'] and kwargs['ax']:
                fig, ax = kwargs['fig'], kwargs['ax']
            else:
                fig, ax = plt.subplots()
            plt.plot(volume_qe, energy_qe,'o', label='DFT',mec=kwargs.get('mec','b'),mfc=kwargs.get('mfc','none'),)
            # if kwargs['txt'] == True:
                # plt.text(0.4,0.6,f"DFT\n"+r"$V_0$"+f"={v0qe:.4f}"+r"$\AA^3$"+f"\n{lattice}={a:.4f}Å\n",transform=ax.transAxes,wrap=True)
            if kwargs.get('close',True):
                plt.close(fig=fig)          
    return None