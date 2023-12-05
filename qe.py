from MLIP_processing import utils
from calctest.espresso import QETest
import numpy as np
import os
from ase.io import read,write
from ase.io.espresso import write_espresso_in
import dpdata

class QEinput(QETest):
    def __init__(
        self,
        cluster ='bridges',
        calculation ='scf',
        input_data = None,
        pseudo_dir = None,
        elements: list=['C','F'],
        ecutwfc = 100,
        pseudopotentials = None,
        ncores = None,
        command = None,
        kspacing = 0.18,
        kpts_list = None,
        atoms_list = None,
        system = None,
        **kwargs,
    ):
        self.calculation = calculation
        # default_input_data = self.default_input()
        if cluster == 'bridges':
            default_input_data = self.input_bridges(self.calculation,pseudo_dir,)
            use_srun = False
            use_mpirun = True
        if cluster == 'arjuna':
            default_input_data = self.input_arjuna(self.calculation,pseudo_dir,)
            use_srun = True
            use_mpirun = False
        self.input_data = default_input_data
        self.input_data['ecutwfc'] = ecutwfc
        if input_data:
            self.input_data.update(input_data)
        self.pseudopotentials = dict()
        for element in elements:
            self.pseudopotentials[element] = f"{element}_ONCV_PBE-1.0.upf"
        if pseudopotentials:
            self.pseudopotentials = pseudopotentials
        
        self.kspacing = kspacing
        self.kpts_list = kpts_list

        calc_params = {
            'input_data': self.input_data,
            'pseudopotentials': self.pseudopotentials,
            'kspacing': self.kspacing,
        }
        super(QEinput, self).__init__(calc_params)
        # self.calc_params = calc_params
        if atoms_list:
            self.atoms_list = atoms_list
        elif system:
            self.atoms_list = self.get_atoms_list(system)
        else:
            raise RuntimeError('No atoms specified!')
    
    def default_input(self):
        default_input = {
            'calculation': 'scf',
            'nstep': 1,
            'ibrav': 0,
            'conv_thr': 1.0e-6,
            'forc_conv_thr': 1.0e-3,
            'etot_conv_thr': 1.0e-4,
            'disk_io': 'nowf',
            'tprnfor': True,
            'tstress': True,
            'electron_maxstep': 200,
        }
        return default_input

    def input_bridges(self, calculation, pseudo_dir=None,):
        input_data = self.default_input()
        if not pseudo_dir:
            pseudo_dir = '/jet/home/hzhao3/pseudopotential/SG15_ONCV_v1.0_upf'
        if calculation == 'scf':
            input_data['calculation']='scf'
        elif calculation == 'relax':
            input_data['calculation']='relax'
            input_data['nstep']=200
            input_data['ion_dynamics'] = 'bfgs'
        elif calculation == 'vc-relax':
            input_data['calculation'] = 'vc-relax'
            input_data['nstep'] = 200
            input_data['ion_dynamics'] = 'bfgs'
            input_data['cell_dynamics'] = 'bfgs'
            input_data['press_conv_thr'] = 0.5
        input_data['pseudo_dir'] = pseudo_dir
        return input_data
    
    def input_arjuna(self, calculation,pseudo_dir=None,):
        if not pseudo_dir:
            pseudo_dir = '/home/hanchen2/pseudopotential_orbital/SG15_ONCV_v1.0_upf/'
        input_data = self.input_bridges(
            calculation,
            pseudo_dir = pseudo_dir,
            )
        return input_data
            
    def get_atoms_list(self, system):
        return system.to('ase/structure')

    def InitJobGen(
            self,
            kpoints_list = None,
            to_dir = None,
            index_range = None,
            **kwargs
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
        kpoints_list: list
                specify kpoints, will ignore kspacing if this parameter defined
        to_dir: str
            parent directory to write input file to

        Returns
        -------
        None
        """

        atoms_list = self.atoms_list
        if not kpoints_list:
            kpoints_list = utils.get_kpoints(kspacing = self.kspacing,atoms_list=self.atoms_list)
        if not index_range:
            index_range = range(len(atoms_list))

        for i in index_range:
            if not f"/{i}/pbe" in to_dir:
                dir = to_dir+f"/{i}/pbe"
            else:
                dir = to_dir
            if not os.path.exists(dir):
                os.makedirs(dir)
            input_data = self.input_data
            pseudopotentials = self.pseudopotentials
            input_data['nat'] = len(atoms_list[i])
            input_data['ntyp'] = len(atoms_list[i].get_atomic_numbers())
            input_data.update(kwargs)
            write_espresso_in(open(dir+'/input','w'), atoms = atoms_list[i],
                            input_data = input_data,
                            pseudopotentials = pseudopotentials,
                            kpts = kpoints_list[i],
                            **kwargs
                            )
        return None

    def EOSinput(
        self,
        shape='hexagonal',
        atoms=None,
        vrange=[0.95,1.05],
        num=5,
        to_dir=None,
        **kwargs
        ):
        if not atoms:
            raise RuntimeError("Need to specify atoms!")
        scale = np.linspace(vrange[0],vrange[1],num)
        if shape=='hexagonal':
            print("In hexagonal cell, a and c EOS are fitted respectively!")
            # fit a EOS
            atoms_list=[]
            if not os.path.exists(to_dir+f'/a/'):
                os.makedirs(to_dir+f'/a/')
            if not os.path.exists(to_dir+f'/c/'):
                os.makedirs(to_dir+f'/c/')
            for i in range(num):
                cell = atoms.get_cell()
                atoms_copy = atoms.copy()
                cell[0] = cell[0]*scale[i]
                cell[1] = cell[1]*scale[i]
                atoms_copy.set_cell(cell,scale_atoms=True)
                atoms_list.append(atoms_copy)
            self.atoms_list = atoms_list
            self.InitJobGen(to_dir=to_dir+f"/a/")
            # fit c EOS
            atoms_list=[]
            for i in range(num):
                cell = atoms.get_cell()
                atoms_copy = atoms.copy()
                cell[2] = cell[2]*scale[i]
                atoms_copy.set_cell(cell,scale_atoms=False)
                atoms_list.append(atoms_copy)
            self.atoms_list = atoms_list
            self.InitJobGen(to_dir=to_dir+f"/c/", **kwargs)
        return None

    def write_sh(self,input_data=None):
        pass