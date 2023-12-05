from MLIP_processing import utils
from MLIP_processing.post_processing.model_test import ModelOut

# from calctest.deepmd import DeepMDTest
from calctest.calctest import CalcTest
import numpy as np
import os
from ase.io import read, write
import dpdata
import matplotlib.pyplot as plt
import logging
import pandas as pd
from ase.units import GPa
import subprocess
from os.path import join
from ase.units import kJ
import glob
import shutil
from ase.eos import EquationOfState
from ase.utils.forcecurve import fit_images
from ase.neb import NEB
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
import warnings


class MultiModelOut:
    # class for evaluating uncertainty of multiple models
    def __init__(
        self,
        models,
        model_ensemble_paths,
        parent_dir,
        test_data,
        DFTcalctest_path,
    ):
        self.models = models
        self.model_ensemble_paths = model_ensemble_paths
        self.parent_dir = parent_dir
        self.test_data = test_data
        self.DFTcalctest_path = DFTcalctest_path
        # self.calc_path = calc_path
        # for i, model_path in enumerate(model_ensemble_paths):
        #     # child_dir is the directory of the model and its post-processing results
        #     if 'pb' in model_path or 'pth' in model_path:
        #         model_folder = model_path.removesuffix('best_model.pth').removesuffix('best_model.pb')
        #     child_dir = join(self.parent_dir,model_folder)
        #     self.models.append(
        #         ModelOut(
        #             test_data = self.test_data,
        #             calc_path = join(child_dir,model_path)
        #             parent_dir = child_dir,
        #             work_path = './',
        #             parity_dir = join(child_dir,'calctest/parity/'),
        #             eos_dir = join(child_dir,'calctest/eos/'),
        #             elastic_dir = join(child_dir,'calctest/elastic/'),
        #             DFTcalctest_path = self.DFTcalctest_path,
        #         )
        #     )

    def MultiEOS(
        self,
        cell_list,
        structure,
        eos_path="calctest/EOS/",
        ranger=[0.95, 1.05],
        num_mlip=25,
        lattice=["a", "c"],
        DFTcalctest_path=None,
        min_volume=False,
        **kwargs,
    ):
        df_list = [dict() for i in range(len(self.model_ensemble_paths))]
        energy = np.empty((len(self.model_ensemble_paths), num_mlip))

        if not DFTcalctest_path:
            DFTcalctest_path = self.DFTcalctest_path

        # # Link DFT calctest to the folder
        # os.chdir(self.parent_dir)
        # if not os.path.exists("./QEcalctest"):
        #     logging.info(f"Linking to QE calctest at {self.parent_dir}")
        #     subprocess.run(["ln", "-s", f"{self.QEcalctest_path}", "./"])
        for i, model_path in enumerate(self.model_ensemble_paths):
            Modelobj = self.models[i]
            os.chdir(Modelobj.parent_dir)
            df_list[i] = Modelobj.EOS(
                cell_list=cell_list,
                eos_path=join(Modelobj.parent_dir, eos_path),
                structure=structure,
                lattice = lattice,
                ranger=ranger,
                num=num_mlip,
                DFTcalctest_path=DFTcalctest_path,
                min_volume=min_volume,
                **kwargs,
            )
            plt.close()
        # Use a dictionary to store every csv file
        eos_dict = dict()
        for stru in structure:
            for supercell in cell_list:
                for lat in lattice:
                    eos_single = dict()
                    eos_single["Volume"] = df_list[0][f"{stru}_{supercell}_{lat}"][
                        "Volume"
                    ]
                    for i, model_path in enumerate(self.model_ensemble_paths):
                        eos_single[
                            f"{model_path.removeprefix(self.parent_dir)}"
                        ] = df_list[i][f"{stru}_{supercell}_{lat}"]["Energy"]
                        energy[i, :] = df_list[i][f"{stru}_{supercell}_{lat}"]["Energy"]
                    eos_single["Mean"] = np.mean(energy, axis=0)
                    eos_single["Std"] = np.std(energy, axis=0)
                    eos_single["Max"] = np.max(energy, axis=0)
                    eos_single["Min"] = np.min(energy, axis=0)
                    eos_single = pd.DataFrame(eos_single)
                    if not os.path.exists(
                        join(self.parent_dir, eos_path)
                        + f"/{stru}/MLIP/{supercell}/{lat}/"
                    ):
                        os.makedirs(
                            join(self.parent_dir, eos_path)
                            + f"/{stru}/MLIP/{supercell}/{lat}/"
                        )
                    eos_single.to_csv(
                        join(self.parent_dir, eos_path)
                        + f"/{stru}/MLIP/{supercell}/{lat}/{self.models[i].calc_name}_{ranger[0]}-{ranger[1]}.csv"
                    )
                    eos_dict[f"{stru}_{supercell}_{lat}"] = eos_single

                    fig, ax = plt.subplots()
                    try:
                        eos = EquationOfState(eos_single["Volume"], eos_single["Mean"])
                        v0, e0, B = eos.fit()
                        B = B / kJ * 1.0e24
                        eos.plot(
                            join(self.parent_dir, eos_path)
                            + f"/{stru}/MLIP/{supercell}/{lat}/{self.models[i].calc_name}_{ranger[0]}-{ranger[1]}.png",
                            ax=ax,
                            color="r",
                            label="MLIP mean",
                            markercolor="r",
                            mec="r",
                            mfc="r",
                        )
                    except:
                        plt.plot(eos_single['Volume'],eos_single['Mean'],'o',mec=kwargs.get('mec','r'),mfc=kwargs.get('mfc','r'),markersize=kwargs.get('markersize',5))
                        warnings.warn(f'EOS for {stru}/{supercell}/{lat} is not possible')
                    if kwargs.get("txt", None):
                        plt.text(
                            0.4,
                            0.8,
                            f"{Modelobj.calc_name}\n"+r'$V_0$'+f"={v0:.4f}"+r"$\AA^3$"+"\n",
                            transform=ax.transAxes,
                            wrap=True,
                        )

                    [num, start] = [kwargs.get("num", 5), kwargs.get("start", 2)]
                    utils.qe_eos(
                        path=join(
                            self.DFTcalctest_path, f"EOS/{stru}/{supercell}/{lat}/"
                        ),
                        num_dft=kwargs.get("num_dft", 5),
                        start=start,
                        lattice=lat,
                        fig=fig,
                        ax=ax,
                        txt=True,
                        plot_path=self.parent_dir
                        + f"calctest/EOS/{stru}/MLIP/{supercell}/{lat}/{self.models[i].calc_name}_{ranger[0]}-{ranger[1]}.png",
                        color="blue",
                        fus=kwargs.get("fus", "C1F1"),
                        mec="b",
                        mfc="none",
                        ls="--",
                        close=False,
                    )
                    emin = eos_single["Min"]
                    emax = eos_single["Max"]
                    v = eos_single["Volume"]
                    ax.fill_between(
                        v,
                        emin,
                        emax,
                        color="r",
                        alpha=0.3,
                        label=f"{self.models[i].calc_name} Uncertainty",
                    )
                    plt.legend()
                    if not os.path.exists(join(self.parent_dir, eos_path)):
                        os.makedirs(join(self.parent_dir, eos_path))
                    # ax.set_xlim([np.min(v),np.max(v)])
                    plt.savefig(
                        join(self.parent_dir, eos_path)
                        + f"/{stru}/MLIP/{supercell}/{lat}/{self.models[i].calc_name}_{ranger[0]}-{ranger[1]}.png",
                        bbox_inches="tight",
                    )
                    plt.close(fig=fig)

    def MultiNEB(
        self,
        run_neb=False,
        mlip_interp_csv_list=None,
        dft_interp_csv=None,
        dft_csv=None,
        mlip_csv=None,
        mlip_path_csv=None,
        savefig_path=None,
        save_csv_path=None,
        **kwargs,
    ):
        #  run_neb determines whether NEB runs will be done
        #  It can be a list containing which models need to run NEB
        if run_neb:
            raise NotImplementedError
        else:
            mlip_interp_df = pd.DataFrame()
            # if run_neb=False, the mlip path interpolation files should be provided
            for i, csv in enumerate(mlip_interp_csv_list):
                df = pd.read_csv(csv)
                mlip_interp_df["index"] = df["index"]
                mlip_interp_df[
                    f"energy_{i}"
                ] = df["energy"]
            mlip_interp_df["Mean"] = mlip_interp_df.iloc[
                :, 1 : len(self.model_ensemble_paths) + 1
            ].mean(axis=1)
            mlip_interp_df["Min"] = mlip_interp_df.iloc[
                :, 1 : len(self.model_ensemble_paths) + 1
            ].min(axis=1)
            mlip_interp_df["Max"] = mlip_interp_df.iloc[
                :, 1 : len(self.model_ensemble_paths) + 1
            ].max(axis=1)
        # with the csv files read into mlip_interp_df, we can obtain the minimum and maximum
        # Here for visualization we also need to specify which plot to show
        # if show_model=None then the path to show is the mean
        if not mlip_csv:
            energies = mlip_interp_df["Mean"]
            path = mlip_interp_df["index"]
        else:
            mlip_df = pd.read_csv(mlip_csv)
            path, energies = mlip_df["index"], mlip_df["energy"]
            assert len(path) == len(energies)
        # read dft csv and dft_interp_csv
        dft_df = pd.read_csv(dft_csv)
        dft_df_interp = pd.read_csv(dft_interp_csv)
        dft_path, dft_energy, dft_path_interp, dft_energy_interp = (
            dft_df["index"],
            dft_df["energy"],
            dft_df_interp["index"],
            dft_df_interp["energy"],
        )
        assert len(dft_path) == len(dft_energy)
        assert len(dft_path_interp) == len(dft_energy_interp)
        # NEB plotting
        fig, ax = plt.subplots()
        # plot the transition path
        plt.plot(
            path,
            energies,
            marker="^",
            markersize=10,
            linestyle="None",
            mec="r",
            mfc="r",
        )
        df = pd.read_csv(mlip_path_csv)
        plt.plot(
            df["index"],
            df["energy"],
            linestyle="-",
            linewidth=2,
            color="r",
            label=f'{self.models[0].calc_name.removesuffix("0")}',
        )
        # Plot uncertainty as max-min
        ax.fill_between(
            mlip_interp_df["index"],
            mlip_interp_df["Min"],
            mlip_interp_df["Max"],
            color="r",
            alpha=0.3,
            label=f'{self.models[0].calc_name.removesuffix("0")} Uncertainty',
        )
        # plot dft
        plt.plot(
            dft_path,
            dft_energy,
            marker="o",
            markersize=10,
            mec="b",
            mfc="none",
            color="b",
            linestyle="None",
        )
        plt.plot(
            dft_path_interp,
            dft_energy_interp,
            ls="--",
            linewidth=2,
            color="b",
            label=f"DFT",
        )
        plt.text(
            0.6,
            0.05,
            f'{self.models[0].calc_name} Mean Barrier = {np.max(mlip_interp_df["Mean"])-np.min(mlip_interp_df["Mean"]):.3f} eV/CF\nDFT Barrier = {np.max(dft_energy_interp)-np.min(dft_energy_interp):.3f} eV/CF',
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.legend()
        plt.xticks([0, 1], labels=[kwargs["start"], kwargs["end"]])
        plt.ylabel("Energy (eV)")
        ax.set_xlim([-0.1, 1.1])
        plt.savefig(savefig_path, bbox_inches="tight")
        mlip_interp_df.to_csv(save_csv_path)
        plt.close(fig=fig)

    # def MultiRDF(
    #     self,
    #     run_rdf=False,
    #     cell_relax=False,
    #     atoms_rdf_csvs=None,
    #     cell_rdf_csvs=None,
    #     **kwargs
    # ):
    #     if run_rdf:
    #         raise NotImplementedError('Please calculate rdf in advance')

    #     if not cell_relax:
    #         for i,traj in atoms_relax_trajs:
    #             atoms=read(traj)
    #             rdf
