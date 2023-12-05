# from MLIP_processing import utils
from calctest.deepmd import DeepMDTest
from MLIP_processing.post_processing.model_test import ModelOut
from MLIP_processing.post_processing.multi_model_test import MultiModelOut
# import numpy as np
import os
# from ase.io import read, write
# import dpdata
# import matplotlib.pyplot as plt
# import logging
# import pandas as pd
# from ase.units import GPa
# import subprocess
# from os.path import join
# from ase.units import kJ
# import glob
# import shutil
# from ase.eos import EquationOfState


class DPout(ModelOut,DeepMDTest):
    def __init__(
        self,
        test_data,
        calc_path,
        parent_dir,
        **kwargs,
    ):
        work_path=kwargs.get('work_path',"./")
        parity_dir=kwargs.get('parity_dir',"./calctest/parity/")
        eos_dir=kwargs.get('eos_dir',"./calctest/eos/")
        elastic_dir=kwargs.get('elastic_dir',"./calctest/elastic/")
        DFTcalctest_path=kwargs.get('DFTcalctest_path',None)
        self.calc_path = calc_path
        self.calc_params = {
            "calc_path": self.calc_path,
        }
        self.work_path = work_path        
        super(DPout, self).__init__(
            calc_params = self.calc_params,
            test_data = test_data,
            calc_path = calc_path, 
            work_path = self.work_path,
            parity_dir = parity_dir,
            parent_dir = parent_dir,
            eos_dir = eos_dir,
            elastic_dir = elastic_dir,
            DFTcalctest_path = DFTcalctest_path,
            calc_name = kwargs.get('calc_name','DP'),
            )
        print(self.calc_name)


class MultiDPout(MultiModelOut):
    # class for calculating uncertainty of multi models
    def __init__(
        self,
        rnd_seed_list,
        parent_dir,
        test_data,
        DFTcalctest_path,
        **kwargs,
    ):
        models = []
        for i,rnd_seed in enumerate(rnd_seed_list):
            calc_name = kwargs.get('calc_name_list',[f"DP{i}" for rnd_seed in rnd_seed_list])[i]
            models.append(
                DPout(
                    test_data = test_data,
                    calc_path = os.path.join(parent_dir,f'{rnd_seed}/best_model.pb'),
                    parent_dir = os.path.join(parent_dir,f'{rnd_seed}/'),
                    calc_name = calc_name,
                    **kwargs,
                )
            )
        super().__init__(
            models = models,
            model_ensemble_paths = [os.path.join(parent_dir,rnd_seed) for rnd_seed in rnd_seed_list],
            parent_dir = parent_dir,
            test_data = test_data,
            DFTcalctest_path = DFTcalctest_path,
            )    
 

    # def MultiEOS(
    #     self,
    #     cell_list=["primitive_cell"],
    #     eos_path="/calctest/EOS/",
    #     structure=["boat", "chair"],
    #     ranger=[0.95, 1.05],
    #     num_dp=25,
    #     lattice=["a", "c"],
    #     QEcalctest_path=None,
    #     min_volume=False,
    #     **kwargs,
    # ):
    #     df_list = [dict() for i in range(len(self.rnd_seed_list))]
    #     energy = np.empty((len(self.rnd_seed_list), num_dp))

    #     if not QEcalctest_path:
    #         QEcalctest_path = self.QEcalctest_path

    #     # Link QE calctest to the folder
    #     os.chdir(self.parent_dir)
    #     if not os.path.exists("./QEcalctest"):
    #         logging.info(f"Linking to QE calctest at {self.parent_dir}")
    #         subprocess.run(["ln", "-s", f"{self.QEcalctest_path}", "./"])

    #     for i in range(len(self.rnd_seed_list)):
    #         DPobj = self.dp[i]
    #         os.chdir(DPobj.parent_dir)
    #         df_list[i] = DPobj.EOS(
    #             cell_list=cell_list,
    #             eos_path=self.parent_dir + f"/{self.rnd_seed_list[i]}/" + eos_path,
    #             structure=structure,
    #             ranger=ranger,
    #             num_dp=num_dp,
    #             lattice=lattice,
    #             QEcalctest_path=QEcalctest_path,
    #             min_volume=False,
    #             **kwargs,
    #         )
    #         plt.close()
    #         # Use a dictionary to store every csv file
    #     eos_dict = dict()
    #     for stru in structure:
    #         for supercell in cell_list:
    #             for lat in lattice:
    #                 eos_single = dict()
    #                 eos_single["Volume"] = df_list[0][f"{stru}_{supercell}_{lat}"][
    #                     "Volume"
    #                 ]
    #                 for i in range(len(self.rnd_seed_list)):
    #                     eos_single[f"{self.rnd_seed_list[i]}"] = df_list[i][
    #                         f"{stru}_{supercell}_{lat}"
    #                     ]["Energy"]
    #                     energy[i, :] = df_list[i][f"{stru}_{supercell}_{lat}"]["Energy"]
    #                 eos_single["Mean"] = np.mean(energy, axis=0)
    #                 eos_single["Std"] = np.std(energy, axis=0)
    #                 eos_single["Max"] = np.max(energy, axis=0)
    #                 eos_single["Min"] = np.min(energy, axis=0)
    #                 eos_single = pd.DataFrame(eos_single)
    #                 eos_single.to_csv(
    #                     self.parent_dir + eos_path + f"/{stru}_{supercell}_{lat}.csv"
    #                 )
    #                 eos_dict[f"{stru}_{supercell}_{lat}"] = eos_single

    #                 fig, ax = plt.subplots()
    #                 eos = EquationOfState(eos_single["Volume"], eos_single["Mean"])
    #                 v0, e0, B = eos.fit()
    #                 B = B / kJ * 1.0e24
    #                 eos.plot(
    #                     self.parent_dir + eos_path
    #                     + f"/{stru}_{supercell}_{lat}_{ranger[0]}-{ranger[1]}.png",
    #                     ax=ax,
    #                     color="r",
    #                     label="MLIP mean",
    #                     markercolor="r",
    #                 )
    #                 plt.text(
    #                     0.4,
    #                     0.8,
    #                     f"DeepMD\nB={B:.4f} GPa\nv0={v0:.4f}Å^3\n",
    #                     transform=ax.transAxes,
    #                     wrap=True,
    #                 )

    #                 if "num" in kwargs:
    #                     num = kwargs["num"]
    #                 else:
    #                     num = 5
    #                 if "start" in kwargs:
    #                     start = kwargs["start"]
    #                 else:
    #                     start = 2
    #                 utils.qe_eos(
    #                     path=self.parent_dir
    #                     + f"/{self.rnd_seed_list[i]}/calctest/EOS/{stru}/qe/{supercell}/{lat}/",
    #                     num=num,
    #                     start=start,
    #                     lattice=lat,
    #                     fig=fig,
    #                     ax=ax,
    #                     txt=True,
    #                     plot_path=self.parent_dir
    #                     + f"/{stru}_{supercell}_{lat}_{ranger[0]}-{ranger[1]}.png",
    #                     color="blue",
    #                 )
    #                 emin = eos_single["Min"]
    #                 emax = eos_single["Max"]
    #                 v = eos_single["Volume"]
    #                 ax.fill_between(
    #                     v, emin, emax, color="r", alpha=0.3, label="Uncertainty"
    #                 )
    #                 plt.legend()
    #                 if not os.path.exists(self.parent_dir+eos_path):
    #                     os.makedirs(self.parent_dir+eos_path)
    #                 # ax.set_xlim([np.min(v),np.max(v)])
    #                 plt.savefig(
    #                     self.parent_dir + eos_path
    #                     + f"/{stru}_{supercell}_{lat}_{ranger[0]}-{ranger[1]}.png",
    #                     bbox_inches="tight",
    #                 )
    #                 plt.close()

    # def MultiNEB(
    #     self,
    #     initial_file,
    #     final_file,
    #     nimages=5,
    #     fmax=0.1,
    #     # fixed_size=4,
    #     struct_name=None,
    #     plot=True,
    #     workdir="./calctest/NEB/MLIP/",
    #     run_name="boat_lattice",
    #     max_steps=400,
    #     relax_initial=False,
    #     relax_final=False,
    #     contraint=False,
    #     **kwargs,
    # ):  
    #     rnd_seed_list = self.rnd_seed_list
    #     # initial and final file should be ase-readable format
    #     initial = read(initial_file)
    #     final = read(final_file)
    #     if 'initial_pbc' in kwargs:
    #         initial.set_pbc(kwargs['initial_pbc'])
    #     else:
    #         initial.set_pbc((1,1,1))
    #     if 'final_pbc' in kwargs:
    #         initial.set_pbc(kwargs['final_pbc'])        
    #     else:
    #         final.set_pbc((1,1,1))

        
        
    #     energy_list = [pd.DataFrame() for i in self.rnd_seed_list]

    #     return None

    # def MultiParity(
    #     self,
    #     test_data=None,
    #     parity_dir=None,
    #     step = 2000,
    #     to_csv=True,
    # ):
    #     if not test_data:
    #         test_data = self.test_data
    #     if not parity_dir:
    #         parity_dir = self.parent_dir + "/calctest/parity/"
    #     if not os.path.exists(parity_dir):
    #         os.makedirs(parity_dir)
    #     print('start parity')
    #     images = read(f"{test_data}@:-1")
    #     en = np.empty((len(self.rnd_seed_list), len(images)))
    #     fc = []
    #     df_e = pd.DataFrame(dict())
    #     df_f = pd.DataFrame(dict())

    #     df_e["dft"] = np.array(
    #         [image.get_potential_energy() / len(image) * 2 for image in images]
    #     )
    #     df_f["dft"] = np.hstack([image.get_forces().flatten() for image in images])
    #     for i in range(len(self.rnd_seed_list)):
    #         DPobj = self.dp[i]

    #         DPobj.parity(
    #             parity_dir = None,
    #             parent_dir = self.parent_dir+f'/{self.rnd_seed_list[i]}/',
    #             step = step,
    #         )
    #         energy=np.empty(len(images))
    #         forces=np.array([])
    #         for j in range(len(images)):
    #             images[j].calc = DPobj.calc
    #             energy[j]=images[j].get_potential_energy()/len(images[j])*2
    #             forces = np.hstack((forces,images[j].get_forces().flatten()))
    #         en[i,:] = energy
    #         fc.append(forces)
    #         df_e[self.rnd_seed_list[i]]=energy
    #         df_f[self.rnd_seed_list[i]]=forces

    #     fc = np.array(fc)
    #     df_e['mean']=np.mean(en,axis=0)
    #     df_e['max']=np.max(en,axis=0)
    #     df_e['min']=np.min(en,axis=0)

    #     df_f['mean']=np.mean(fc,axis=0)
    #     df_f['max'] = np.max(fc,axis=0)
    #     df_f['min']=np.min(fc,axis=0)

    #     if to_csv:
    #         df_e.to_csv(parity_dir+'/energy.csv')
    #         df_f.to_csv(parity_dir+'/forces.csv')

    #     # df_e = pd.read_csv(parity_dir + "/energy.csv")
    #     # df_f = pd.read_csv(parity_dir + "/forces.csv")
    #     df_e = df_e.sort_values(by=['dft'])
    #     df_f = df_f.sort_values(by=['dft'])
    #     # if to_csv:
    #     #     df_e.to_csv(parity_dir+'/energy.csv')
    #     #     df_f.to_csv(parity_dir+'/forces.csv')        
    #     fig, ax = plt.subplots()
    #     xlim = [
    #         np.min(df_e["dft"]) - 0.1 * (np.max(df_e["dft"]) - np.min(df_e["dft"])),
    #         np.max(df_e["dft"]) + 0.1 * (np.max(df_e["dft"]) - np.min(df_e["dft"])),
    #     ]
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(xlim)
    #     plt.plot(xlim, xlim, "--", color="k")
    #     rmse = np.sqrt(np.sum(np.square(df_e['dft']-df_e['mean']))/len(df_e['mean']))
    #     plt.plot(df_e["dft"], df_e["mean"], "o", color="r",label=f'RMSE={np.round(rmse, 2)*1000}eV/atom')
    #     ax.fill_between(
    #         df_e["dft"],
    #         df_e["min"],
    #         df_e["max"],
    #         color="r",
    #         alpha=0.4,
    #         interpolate=True,
    #         label="Uncertainty",
    #     )
    #     plt.legend()
    #     plt.xlabel("DFT Energy per CF (eV)")
    #     plt.xlabel("MLIP Energy per CF (eV)")
    #     plt.savefig(parity_dir + "/energy.png", bbox_inches="tight")
    #     plt.close()

    #     fig, ax = plt.subplots()
    #     xlim = [np.min(df_f["dft"]) * 1.005, np.max(df_f["dft"]) * 0.995]
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(xlim)
    #     plt.plot(xlim, xlim, "--", color="k")
    #     rmse = np.sqrt(np.sum(np.square(df_f['dft']-df_f['mean']))/len(df_f['mean']))
    #     plt.plot(df_f["dft"], df_f["mean"], ".", color="r",label=f'RMSE={np.round(rmse, 2)*1000}meV/'+r'$\AA$')
    #     ax.fill_between(
    #         df_f["dft"],
    #         df_f["min"],
    #         df_f["max"],
    #         color="r",
    #         alpha=0.4,
    #         interpolate=True,
    #         label="Uncertainty",
    #     )
    #     plt.legend()
    #     plt.xlabel("DFT Energy per CF (eV)")
    #     plt.xlabel("MLIP Energy per CF (eV)")
    #     plt.savefig(parity_dir + "/forces.png", bbox_inches="tight")
    #     plt.close()
