from MLIP_processing import utils

# from calctest.deepmd import DeepMDTest
from calctest.calctest import CalcTest
import numpy as np
import math
import os
from ase.io import read, write

# import dpdata
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
from ase.QEneb import QENEB
from ase.neb import NEB, NEBTools
from ase.optimize import BFGS, GPMin, MDMin
from ase.io.trajectory import Trajectory
from ase.geometry.analysis import Analysis
from ase import Atoms


class ModelOut(CalcTest):
    def __init__(
        self,
        test_data,
        calc_params,
        parent_dir,
        work_path="./",
        parity_dir="./calctest/parity/",
        eos_dir="./calctest/eos/",
        elastic_dir="./calctest/elastic/",
        DFTcalctest_path=None,
        **kwargs,
    ):
        self.calc_params = calc_params
        self.calc_path = self.calc_params.get("calc_path", None)
        self.work_path = work_path
        super(ModelOut, self).__init__(
            calc_params=self.calc_params,
            work_path=self.work_path,
            calc_name=kwargs.get("calc_name", "Calc"),
        )
        self.test_data = test_data
        self.parity_dir = parity_dir
        self.parent_dir = parent_dir
        self.eos_dir = eos_dir
        self.elastic_dir = elastic_dir
        self.DFTcalctest_path = DFTcalctest_path

    def parity(
        self,
        parity_dir=None,
        parent_dir=None,
        step=2000,
        lcurve=True,
        lcurvepath="./lcurve.out",
        savefig=True,
        show=False,
        **kwargs,
    ):
        if not parent_dir:
            parent_dir = self.parent_dir
        os.chdir(parent_dir)
        test_data = self.test_data
        if not parity_dir:
            parity_dir = self.parity_dir
        self._make_workdir(parity_dir)
        # print('parity_dir: ',parity_dir)
        if not os.getenv("PWD") == parent_dir:
            os.chdir(parent_dir)
        res = self.parity_plot(
            data=test_data,
            forces=kwargs.get("forces", True),
            force_prob=1.0,
            stress=kwargs.get("stress", True),
            n_cores=1,
            struct_name="",
            unit="meV",
            index=":",
            workdir=parity_dir,
            fontsize=10,
        )
        fig, ax = plt.subplots()
        os.chdir(parent_dir)
        # print(lcurve)
        if lcurve == True:
            utils.plot_loss_curve(
                path=lcurvepath,
                step=step,
                savefig=savefig,
                figpath=parity_dir,
                show=show,
            )
        plt.close(fig=fig)
        logging.info("Parity Plot Done")
        os.chdir(parent_dir)
        del res
        # return df

    def EOS(
        self,
        cell_list=["primitive_cell"],
        eos_path="calctest/EOS/",
        structure=["boat", "chair"],
        ranger=[0.95, 1.05],
        num=25,
        lattice=["a", "c"],
        DFTcalctest_path=None,
        min_volume=False,
        csv_path=None,
        fus=None,
        **kwargs,
    ):
        parent_dir = self.parent_dir
        os.chdir(parent_dir)
        if not eos_path:
            eos_path = join(self.work_path, "", self.eos_dir)
        if not DFTcalctest_path:
            DFTcalctest_path = self.DFTcalctest_path
        for name in structure:
            if not os.path.exists(eos_path + f"/{name}/"):
                os.makedirs(eos_path + f"/{name}/")
                for supercell in cell_list:
                    if not os.path.exists(eos_path + f"/{name}/MLIP/{supercell}"):
                        os.makedirs(eos_path + f"/{name}/MLIP/{supercell}")
                        for lat in lattice:
                            if not os.path.exists(
                                eos_path + f"/{name}/MLIP/{supercell}/{lat}"
                            ):
                                os.makedirs(
                                    eos_path + f"/{name}/MLIP/{supercell}/{lat}"
                                )
            if not os.path.exists(eos_path + f"/{name}/qe/"):
                logging.info("Linking to QEcalctest")
                os.chdir(eos_path + f"/{name}/")
                # print(DFTcalctest_path + f"/EOS/{name}")
                # subprocess.run(["ln", "-s", DFTcalctest_path + f"/EOS/{name}", "./qe"])
                os.chdir(parent_dir)
        logging.info("mkdir done")
        ranger = ranger
        df_dict = dict()
        for stru in structure:
            for supercell in cell_list:
                for lat in lattice:
                    fig, ax = plt.subplots()
                    if lat == "a":
                        a_range = ranger
                        c_range = None
                    elif lat == "c":
                        a_range = None
                        c_range = ranger
                    workdir = eos_path + f"{stru}/MLIP/{supercell}/" + lat + "/"
                    struct_name = ""
                    color = kwargs.get("color", "r")
                    markercolor = "r"
                    
                    res = self.eos(
                        input_file=DFTcalctest_path
                        + f"/EOS/{stru}/{supercell}/init.cif",
                        struct_name=struct_name,
                        num_points=num,
                        v_range=None,
                        a_range=a_range,
                        c_range=c_range,
                        plot=True,
                        workdir=workdir,
                        fig=fig,
                        ax=ax,
                        # txt=kwargs.get('txt',True),
                        markercolor=markercolor,
                        color=color,
                        fus=fus,
                        **kwargs,
                    )
                    os.chdir(parent_dir)
                    plot_path = (
                        workdir
                        + f"/{self.calc_name}_{struct_name}_eos_{ranger[0]:.3f}-{ranger[1]:.3f}.png"
                    )
                    num_dft = kwargs.get("num_dft", 5)
                    start = kwargs.get("start", 2)
                    # print('num_dft',num_dft)
                    utils.qe_eos(
                        path=DFTcalctest_path + f"/EOS/{stru}/{supercell}/{lat}",
                        num_dft=num_dft,
                        start=start,
                        lattice=lat,
                        fig=fig,
                        ax=ax,
                        txt=False,
                        plot_path=plot_path,
                        color="blue",
                        fus=fus,
                        mec="b",
                        mfc="none",
                        ls="--",
                        relative=kwargs.get("relative", None),
                    )
                    if min_volume:
                        min_volume_path = (
                            DFTcalctest_path
                            + f"/relax/{stru}/{supercell}/10kbar/output"
                        )
                        atoms = read(min_volume_path)
                        v = atoms.get_volume()
                        e = atoms.get_potential_energy()
                        plt.axvline(v, 0, 1, linestyle="dashed", label="10kbar")
                        plt.legend()
                        plt.savefig(
                            workdir
                            + f"/min_volume_{ranger[0]:.3f}-{ranger[1]:.3f}.png",
                            bbox_inches="tight",
                        )
                    plt.close(fig=fig)
                    csv_path = (
                        workdir
                        + f"/{self.calc_name}__eos_{ranger[0]:.3f}-{ranger[1]:.3f}.csv"
                    )
                    df = pd.DataFrame({"Volume": res["v"], "Energy": res["e"]})
                    df.to_csv(csv_path)
                    df_dict[f"{stru}_{supercell}_{lat}"] = df
                    os.chdir(parent_dir)
        return df_dict

    def freeze(self, model_name="graph"):
        os.chdir(self.parent_dir)
        subprocess.run(["dp", "freeze", "-o", f"./{model_name}.pb"])
        logging.info("freeze done")

    def mkckpt(self, name="./ckpt", move=True):
        os.chdir(self.parent_dir)
        os.mkdir(name)
        if move:
            filelist = glob.glob("model.ckpt*")
            for file in filelist:
                shutil.move(file, "./ckpt")
        logging.info("move ckpt done")

    # def db(
    #     self,
    #     path,
    #     names=["train", "test", "val"],
    #     chemical_name=None,
    # ):
    #     if not path:
    #         path = (
    #             self.test_data
    #             if "db" not in self.test_data
    #             else self.test_data - "test.db"
    #         )
    #     if not os.path.exists(path + "/deepmd_test"):
    #         raise RuntimeError("Wrong folder!")
    #     for i in range(len(names)):
    #         if not os.path.exists(path + "/" + names[i] + ".db/"):
    #             db = connect(path + "/" + names[i] + ".db/")
    #             system = dpdata.LabeledSystem()
    #             atoms_list = []
    #             for c in len(chemical_name):
    #                 system = dpdata.LabeledSystem(
    #                     f"./deepmd_{names[i]}/{chemical_name[c]}", "deepmd/raw"
    #                 )
    #                 atoms_list.extend(
    #                     dpdata.LabeledSystem(
    #                         f"./deepmd_{names[i]}/{chemical_name[c]}", "deepmd/raw"
    #                     ).to("ase/structure")
    #                 )
    #             for atoms in atoms_list:
    #                 db.write(atoms)

    def elastic_fd(
        self,
        structure="boat",
        supercell="primitive_cell",
        elastic_dir=None,
        C11C12=True,
        C33=True,
        C13=True,
        Ct=True,
        to_csv=True,
        csv_path=None,
        DFTcalctest_path=None,
    ):
        parent_dir = self.parent_dir
        os.chdir(parent_dir)
        # make elastic folder for calculation
        if not elastic_dir:
            elastic_dir = self.elastic_dir
        if not DFTcalctest_path:
            DFTcalctest_path = self.DFTcalctest_path
        self._make_workdir(elastic_dir)
        # if not os.path.exists(elastic_dir + "/qe"):
        #     os.chdir(elastic_dir)
        #     subprocess.run(["ln", "-s", DFTcalctest_path + "/Elastic/", "./qe"])
        #     os.chdir(parent_dir)
        # make sure we are in parent dir
        if not os.getenv("PWD") == parent_dir:
            os.chdir(parent_dir)

        # # Change Elastic qe folder into scale format if not done so before
        # os.chdir(elastic_dir + f"/qe/{structure}/{supercell}")
        # if not os.path.exists("a+0.000_c+0.000"):
        #     for i in range(25):
        #         c = round(np.floor(i % 5) * 0.005 - 0.01, 3)
        #         a = round(np.floor(i / 5) * 0.005 - 0.01, 3)
        #         os.rename(f"./{i:d}", f"./a{a:+.3f}_c{c:+.3f}")
        #         if os.path.exists(f"./{i:d}"):
        #             raise RuntimeError()
        # os.chdir(parent_dir)

        result = {'Properties':[],'MLIP(GPa)':[],'DFT(GPa)':[],'Error(%)':[]}
        self._make_workdir(f'{elastic_dir}/{structure}')
        # calculate C11+C12
        if C11C12:
            # MLIP
            scales = np.linspace(-0.01,0.01,3)
            E = np.zeros(3)
            images = []
            for i,scale in enumerate(scales):
                image = read(f'{self.DFTcalctest_path}/Elastic/{structure}/a-{scale:.2f}_c0.00/input')
                images.append(image)
                image.calc = self.calc
                E[i] = image.get_potential_energy()
                image.write(f'{self.elastic_dir}/{structure}/a-{scale:.2f}_c0.00.xyz')
            cell = images[1].get_cell()
            v0 = images[1].get_volume()
            a0 = cell[0][0]
            C11C12_MLIP = (
                a0**2 / 2 / v0 * (E[2] + E[0] - 2 * E[1]) / (0.01 * a0) ** 2 / GPa
            )
            # DFT
            images = []
            for i,scale in enumerate(scales):
                image = read(f'{self.DFTcalctest_path}/Elastic/{structure}/a-{scale:.2f}_c0.00/output')
                images.append(image)
                E[i] = image.get_potential_energy()
            v0 = images[1].get_volume()
            a0 = images[1].get_cell()[0][0]
            C11C12_DFT = (
                a0**2 / 2 / v0 * (E[2] + E[0] - 2 * E[1]) / (0.01 * a0) ** 2 / GPa
            )
            logging.info("C11C12 done")
            result['Properties'].append('C11C12')
            result['MLIP(GPa)'].append(C11C12_MLIP)
            result['DFT(GPa)'].append(C11C12_DFT)
            result['Error(%)'].append(np.abs(C11C12_MLIP-C11C12_DFT)/C11C12_DFT*100)
        # calculate C33
        if C33:
            # MLIP
            scales = np.linspace(-0.01,0.01,3)
            E = np.zeros(3)
            images = []
            for i,scale in enumerate(scales):
                image = read(f'{self.DFTcalctest_path}/Elastic/{structure}/a-0.00_c{scale:.2f}/input')
                images.append(image)
                image.calc = self.calc
                E[i] = image.get_potential_energy()
                image.write(f'{self.elastic_dir}/{structure}/a-0.00_c{scale:.2f}.xyz')
            cell = images[1].get_cell()
            v0 = images[1].get_volume()
            c0 = cell[2][2]
            C33_MLIP = (
                c0**2 / 2 / v0 * (E[2] + E[0] - 2 * E[1]) / (0.01 * c0) ** 2 / GPa
            )
            print(f'C33 = {C33_MLIP}')
            # DFT
            images = []
            for i,scale in enumerate(scales):
                image = read(f'{self.DFTcalctest_path}/Elastic/{structure}/a-0.00_c{scale:.2f}/output')
                images.append(image)
                E[i] = image.get_potential_energy()
            v0 = images[1].get_volume()
            c0 = images[1].get_cell()[2][2]
            C33_DFT = (
                c0**2 / 2 / v0 * (E[2] + E[0] - 2 * E[1]) / (0.01 * c0) ** 2 / GPa
            )
            logging.info("C33 done")
            result['Properties'].append('C33')
            result['MLIP(GPa)'].append(C33_MLIP)
            result['DFT(GPa)'].append(C33_DFT)
            result['Error(%)'].append(np.abs(C33_MLIP-C33_DFT)/C33_DFT*100)            
        # calculate C13
        if C13:
            # MLIP
            E = np.zeros((2, 2))
            images = [[Atoms()]*2,[Atoms()]*2]
            scales = [-0.01,0.01]
            for i, scale1 in enumerate(scales):
                for j, scale2 in enumerate(scales):
                    image = read(f'{self.DFTcalctest_path}/Elastic/{structure}/a-{scale1:.2f}_c{scale2:.2f}/input')
                    images[i][j] = image
                    image.calc = self.calc
                    E[i,j] = image.get_potential_energy()
                    image.write(f'{self.elastic_dir}/{structure}/a-{scale1:.2f}_c{scale2:.2f}.xyz')                    
            atom = images[1][1]
            v0 = atom.get_volume()
            c0 = atom.get_cell()[-1][-1]
            a0 = atom.get_cell()[0][0]
            C13_MLIP = (
                c0
                * a0
                / 2
                / v0
                * (E[1, 1] + E[0, 0] - E[0, 1] - E[1, 0])
                / 4
                / (0.01 * c0)
                / (0.01 * a0)
                / GPa
            )
            # DFT
            images = [[Atoms()]*2,[Atoms()]*2]
            for i,scale1 in enumerate(scales):
                for j, scale2 in enumerate(scales):
                    image = read(f'{self.DFTcalctest_path}/Elastic/{structure}/a-{scale1:.2f}_c{scale2:.2f}/output')
                    images[i][j]=image
                    E[i,j] = image.get_potential_energy()
            
            v0 = images[1][1].get_volume()
            c0 = images[1][1].get_cell()[2][2]            
            a0 = images[1][1].get_cell()[0][0]
            C13_DFT = (
                c0
                * a0
                / 2
                / v0
                * (E[1, 1] + E[0, 0] - E[0, 1] - E[1, 0])
                / 4
                / (0.01 * c0)
                / (0.01 * a0)
                / GPa
            )
            logging.info("C13 done")
            result['Properties'].append('C13')
            result['MLIP(GPa)'].append(C13_MLIP)
            result['DFT(GPa)'].append(C13_DFT)
            result['Error(%)'].append(np.abs(C13_MLIP-C13_DFT)/C13_DFT*100)               
        # calculate Ct
        if Ct:
            # MLIP
            Ct_MLIP = (C11C12_MLIP + 2 * C33_MLIP - 4 * C13_MLIP) / 6
            # DFT
            Ct_DFT = (C11C12_DFT + 2 * C33_DFT - 4 * C13_DFT) / 6
            logging.info("Ct done")
            result['Properties'].append('Ct')
            result['MLIP(GPa)'].append(Ct_MLIP)
            result['DFT(GPa)'].append(Ct_DFT)
            result['Error(%)'].append(np.abs(Ct_MLIP-Ct_DFT)/Ct_DFT*100)                   
        # df = pd.DataFrame(
        #     {
        #         "Properties": ["C11+C12", "C13", "C33", "Ct"],
        #         "QE": [C11C12_qe, C13_qe, C33_qe, Ct_qe],
        #         "MLIP": [C11C12_MLIP, C13_MLIP, C33_MLIP, Ct_MLIP],
        #         "Error(%)": [
        #             np.abs(C11C12_MLIP - C11C12_qe) / np.abs(C11C12_qe) * 100,
        #             np.abs(C13_MLIP - C13_qe) / np.abs(C13_qe) * 100,
        #             np.abs(C33_MLIP - C33_qe) / np.abs(C33_qe) * 100,
        #             np.abs(Ct_MLIP - Ct_qe) / np.abs(Ct_qe) * 100,
        #         ],
        #     }
        # )
        df = pd.DataFrame(result)
        if to_csv:
            if csv_path == None:
                csv_path = elastic_dir + f"/{structure}/{supercell}.csv"
            df.to_csv(csv_path)

        return None

    def activation_barrier(
        self,
        initial_file,
        final_file,
        fixed_size=4,
        struct_name=None,
        plot_DFT=False,
        workdir=None,
        run_name=None,
        relax_initial=False,
        relax_final=False,
        constraint=False,
        **kwargs,
    ):
        # unwrap kwargs
        images_dft = kwargs.get("images_dft",None)
        dft_csv = kwargs.get('dft_csv',None)
        dft_interp_csv = kwargs.get('dft_interp_csv',None)
        npoints = kwargs.get("npoints", 10)
        fmax = kwargs.get("fmax", 0.1)
        fit = kwargs.get("fit", True)
        plot = kwargs.get("plot", True)
        max_steps = kwargs.get("max_steps", 100)
        DFTcalctest_path = kwargs.get("DFTcalctest_path", None)
        images = kwargs.get("images", None)
        parallel = kwargs.get("parallel", False)
        climb = kwargs.get("climb", False)
        allow_shared_calculator = kwargs.get("allow_shared_calculator", True)

        parent_dir = self.parent_dir
        os.chdir(parent_dir)
        if not DFTcalctest_path:
            DFTcalctest_path = self.DFTcalctest_path
        initial = read(initial_file)
        final = read(final_file)
        if images is None:
            images = [initial]
            for i in range(npoints - 2):
                image = initial.copy()
                image.set_pbc((1, 1, 1))
                image.calc = self.calc
                images.append(image)
            images.append(final)
            images[0].calc = self.calc
            images[-1].calc = self.calc
            variable_k = kwargs.get("variable_k", False)
            if not variable_k:
                neb = NEB(
                    images,
                    parallel=parallel,
                    climb=climb,
                    allow_shared_calculator=allow_shared_calculator,
                    k=kwargs.get("k", 0.1),
                    # method='improvedtangent',
                )
            else:
                neb = QENEB(
                    images,
                    parallel=parallel,
                    climb=climb,
                    allow_shared_calculator=allow_shared_calculator,
                    k=kwargs.get("k", 0.1),
                    kmax=kwargs.get("kmax", 0.62),
                    kmin=kwargs.get("kmin", 0.19),
                    variable_k=variable_k,
                    # method='improvedtangent',
                )
            neb.interpolate()
        else:
            if isinstance(images, str):
                images = read(images)
            for i, image in enumerate(images):
                image.set_pbc((1, 1, 1))
                image.calc = self.calc
            neb = NEB(
                images=images,
                parallel=parallel,
                climb=climb,
                allow_shared_calculator=allow_shared_calculator,
                k=kwargs.get("k", 0.1),
                # kmax=kwargs.get('kmax',0.62),
                # kmin=kwargs.get('kmin',0.19),
                # variable_k = kwargs.get('variable_k',True),
                # method='improvedtangent',
            )
        dyn = BFGS(neb, logfile=workdir + f"/{run_name}/{struct_name}_{fmax}.log")
        traj = Trajectory(
            workdir + f"/{run_name}/{struct_name}_{fmax}.traj",
            "w",
            neb,
            properties=[
                "energy",
                "forces",
            ],
        )
        dyn.attach(traj)
        dyn.run(fmax=fmax, steps=max_steps)
        # res = self.diffusion_barrier(
        #     initial_file=initial_file,
        #     final_file=final_file,
        #     npoints=npoints,
        #     fmax=fmax,
        #     fixed_size=fixed_size,
        #     struct_name=struct_name,
        #     plot=True,
        #     workdir=workdir + f"/{run_name}/",
        #     max_steps=max_steps,
        #     relax_initial=relax_initial,
        #     relax_final=relax_final,
        #     constraint=constraint,
        # )
        with open(workdir + f"/{run_name}/converged.txt", "w") as f:
            f.write(f"Convergence: {dyn.converged()}")
        images = read(workdir + f"/{run_name}/{struct_name}_{fmax}.traj@-{npoints}:")
        write(workdir + f"/{run_name}/{npoints}dyn{fmax:.2f}.xyz", images)
        images = read(workdir + f"/{run_name}/{npoints}dyn{fmax:.2f}.xyz", index=":")
        write(
            workdir + f"/{run_name}/{npoints}dyn{fmax:.2f}.gif", images, rotation="-90x"
        )
        if fit:
            for image in images:
                image.set_pbc((1, 1, 1))
                image.calc = self.calc
            forcefit = fit_images(images, num=20)
            energies = (forcefit.energies + images[0].get_potential_energy()) / len(images[0])* 2
            fit_energies = forcefit.fit_energies
            path = forcefit.path / np.max(forcefit.fit_path)
            fit_path = forcefit.fit_path / np.max(forcefit.fit_path)
            en = (fit_energies + images[0].get_potential_energy()) / len(images[0]) * 2 
            barrier = np.max(fit_energies) / len(images[0]) * 2
            df = pd.DataFrame(
                {
                    "index": fit_path,
                    "energy": en,
                }
            )
            df.to_csv(workdir + f"/{run_name}/{npoints}dyn{fmax:.2f}interp.csv")
            df = pd.DataFrame(
                {
                    "index": path,
                    "energy": energies,
                }
            )
            df.to_csv(workdir + f"/{run_name}/{npoints}dyn{fmax:.2f}.csv")
        else:
            df = pd.DataFrame(
                {"Image": np.arange(0, npoints + 2, 1), "Energy": res["energies"]}
            )
            df.to_csv(workdir + f"/{run_name}/{npoints}dyn{fmax}.csv")
        if plot:
            fig, ax = plt.subplots()
            plt.plot(
                fit_path * npoints,
                en,
                "-",
                color="r",
                label="MLIP",
                linewidth=3,
            )
            plt.plot(
                path * npoints,
                energies,
                "^",
                color=kwargs.get("MLIP_color", "r"),
                markersize=kwargs.get("markersize", 10),
            )
            plt.xticks([0, npoints], labels=[kwargs["start"], kwargs["end"]])
            plt.ylabel("Energy (eV)")
            plt.text(
                0.6,
                0.2,
                f"{self.calc_name} Barrier = {barrier:.3f} eV/CF",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                wrap=True,
            )
            if plot_DFT:
                if not images_dft is None:
                # images_dft = kwargs["images_dft"]
                    forcefit = fit_images(images_dft, num=20)
                    energies = (forcefit.energies + images[0].get_potential_energy()) / len(images_dft[0]) * 2
                    path = forcefit.path / np.max(forcefit.fit_path)
                    fit_energies = forcefit.fit_energies
                    fit_path = forcefit.fit_path / np.max(forcefit.fit_path)
                    en = (fit_energies + images_dft[0].get_potential_energy()) / len(images_dft[0]) * 2
                    barrier = np.max(fit_energies) / len(images_dft[0]) * 2
                    df_dft = pd.DataFrame({"index": fit_path, "energy": en})
                    df_dft.to_csv(workdir + f"/{run_name}/{npoints}images_dft_interp.csv")
                    df_dft = pd.DataFrame({"index": path, "energy": energies})
                    df_dft.to_csv(workdir + f"/{run_name}/{npoints}images_dft.csv")

                    plt.plot(
                        fit_path * npoints,
                        en,
                        "--",
                        color=kwargs.get("DFT_color", "b"),
                        label="DFT",
                        linewidth=3,
                    )
                    plt.plot(
                        path * npoints,
                        energies,
                        "o",
                        mfc="none",
                        mec=kwargs.get("DFT_color", "b"),
                        markersize=kwargs.get("markersize", 10),
                    )
                
                elif dft_csv is not None and dft_interp_csv is not None:
                    df_dft=pd.read_csv(dft_interp_csv)
                    # df_dft['energy'] = df_dft['energy'] * 2
                    df_dft.to_csv(workdir + f"/{run_name}/{npoints}images_dft_interp.csv")
                    plt.plot(
                        df_dft['index'] * npoints,
                        df_dft['energy'],
                        "--",
                        color=kwargs.get("DFT_color", "b"),
                        label="DFT",
                        linewidth=3,
                    )   
                    barrier = np.max(df_dft['energy'])-np.min(df_dft['energy'])

                    df_dft = pd.read_csv(dft_csv)
                    # some silly mistake in original csv file, this multiply 2 should be removed in future
                    # df_dft['energy'] = df_dft['energy'] * 2
                    df_dft.to_csv(workdir + f"/{run_name}/{npoints}images_dft.csv")
                    plt.plot(
                        df_dft['index'] * npoints,
                        df_dft['energy'],
                        "o",
                        mfc="none",
                        mec=kwargs.get("DFT_color", "b"),
                        markersize=kwargs.get("markersize", 10),
                    )

                plt.text(
                    0.6,
                    0.1,
                    f"DFT Barrier = {barrier:.3f} eV/CF",
                    transform=ax.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                    wrap=True,
                )
            ax.set_xlim([-1, npoints + 1])
            plt.legend()
            plt.savefig(
                workdir + f"/{run_name}/{npoints}barrier_fmax={fmax:.2f}.png",
                bbox_inches="tight",
            )
            plt.close(fig=fig)
        return df

    def relax(
        self,
        atoms,
        atoms_ref,
        relax_dir=None,
        fmax=0.02,
        atoms_traj=None,
        rmax=6,
        **kwargs,
    ):
        if isinstance(atoms, str):
            atoms = read(atoms)
        if not relax_dir:
            relax_dir = os.path.join(self.parent_dir, "relax/")
        if not atoms_traj:
            atoms_traj = os.path.join(relax_dir, "relax/atoms_relax.traj")
        if not os.path.exists(relax_dir):
            os.makedirs(relax_dir)
        os.chdir(relax_dir)
        atoms.calc = self.calc
        atoms_opt = BFGS(atoms, logfile="atoms_relax.log")
        traj = Trajectory(
            atoms_traj,
            kwargs.get("file_mode", "w"),
            atoms,
            properties=["energy", "forces", "stress"],
        )
        atoms_opt.attach(traj)
        atoms_opt.run(fmax=fmax,steps=kwargs.get('steps',100))
        images = read(atoms_traj, index=":")

        write(os.path.join(relax_dir, "relax.xyz"), images)
        images = read(os.path.join(relax_dir, "relax.xyz"), index=":")
        write(os.path.join(relax_dir, "relax.gif"), images, rotation="-90x")

        # compare DFT and MLIP RDF
        if isinstance(atoms_ref, str):
            atoms_ref = read(atoms_ref)
        atoms = images[-1]
        if any(atoms.cell[i, i] < 3 * rmax for i in range(3)):
            cell = atoms.cell
            repeat = [int(3 * rmax / cell[i, i]) + 1 for i in range(3)]
            atoms = atoms.repeat((repeat[0], repeat[1], repeat[2]))
        if any(atoms_ref.cell[i, i] < 3 * rmax for i in range(3)):
            cell = atoms_ref.cell
            repeat = [int(3 * rmax / cell[i, i]) + 1 for i in range(3)]
            atoms_ref = atoms_ref.repeat((repeat[0], repeat[1], repeat[2]))
        # print(atoms)
        # print(atoms_ref)
        assert len(atoms) == len(
            atoms_ref
        ), f"Please check if atoms are the same, len(atoms)={len(atoms)},len(atoms_ref)={len(atoms_ref)}"
        temp = np.zeros(3)
        assert all(
            i >= 2 * rmax for i in np.linalg.norm(atoms.cell, axis=1)
        ), "atoms cell < rmax"
        assert all(
            i >= 2 * rmax for i in np.linalg.norm(atoms_ref.cell, axis=1)
        ), "atoms ref cell < rmax"
        anaref = Analysis(atoms_ref)
        ana = Analysis(atoms)
        rdf_ref = anaref.get_rdf(rmax=rmax, nbins=kwargs.get("nbins", math.ceil(rmax*100)))[0]
        rdf = ana.get_rdf(rmax=rmax, nbins=kwargs.get("nbins", math.ceil(rmax*100)))[0]
        fig, ax = plt.subplots()
        plt.plot(
            np.linspace(0, rmax, kwargs.get("nbins", math.ceil(rmax*100))), rdf, label="MLIP", color="r"
        )
        plt.plot(
            np.linspace(0, rmax, kwargs.get("nbins", math.ceil(rmax*100))),
            rdf_ref,
            label="DFT",
            color="b",
            ls="--",
        )
        plt.xlabel("r (" + r"$\AA$" + ")")
        plt.ylabel("g(r)")
        plt.legend()
        plt.savefig(os.path.join(relax_dir, f"rdf{rmax}A.png"), bbox_inches="tight")
        plt.close(fig=fig)
        df = pd.DataFrame(
            {
                "r": np.linspace(0, rmax, kwargs.get("nbins", math.ceil(rmax*100))),
                "rdf": rdf,
                "rdf_dft": rdf_ref,
            }
        )
        df.to_csv(relax_dir + f"/rdf{rmax}A.csv")
        del df
        del images

    def cell_relax(
        self,
        atoms,
        atoms_ref,
        relax_dir=None,
        fmax=0.025,
        cell_traj=None,
        **kwargs,
    ):
        if isinstance(atoms, str):
            atoms = read(atoms)
        if not relax_dir:
            relax_dir = os.path.join(self.parent_dir, "relax/")
        if not cell_traj:
            cell_traj = os.path.join(relax_dir, "relax/cell_relax.traj")
        if not os.path.exists(relax_dir):
            os.makedirs(relax_dir)
        os.chdir(relax_dir)
        atoms.calc = self.calc
        # apply strain filter
        from ase.constraints import StrainFilter
        sf = StrainFilter(atoms)
        atoms_opt = BFGS(sf, logfile="cell_relax.log")
        traj = Trajectory(
            cell_traj,
            kwargs.get("file_mode", "w"),
            atoms,
            properties=["energy", "forces", "stress"],
        )
        atoms_opt.attach(traj)
        atoms_opt.run(fmax=fmax,steps = kwargs.get('max_steps',100))
        images = read(cell_traj, index=":")

        write(os.path.join(relax_dir, "cell-relax.xyz"), images)
        images = read(os.path.join(relax_dir, "cell-relax.xyz"), index=":")
        write(os.path.join(relax_dir, "cell-relax.gif"), images, rotation="-45x")

        # compare DFT and MLIP lattice
        if isinstance(atoms_ref, str):
            atoms_ref = read(atoms_ref)
        atoms = images[-1]
        assert len(atoms) == len(
            atoms_ref
        ), f"Please check if atoms are the same, len(atoms)={len(atoms)},len(atoms_ref)={len(atoms_ref)}"
        df = pd.DataFrame()
        df['param'] = ['a','b','c','<b,c>','<a,c>','<a,b>']
        df['MLIP'] = atoms.get_cell_lengths_and_angles()
        df['DFT'] = atoms_ref.get_cell_lengths_and_angles()
        df['Error'] = np.abs(atoms.get_cell_lengths_and_angles()-atoms_ref.get_cell_lengths_and_angles())
        df['Error(%)'] = np.abs(atoms.get_cell_lengths_and_angles()-atoms_ref.get_cell_lengths_and_angles())/atoms_ref.get_cell_lengths_and_angles()*100
        df.set_index('param')
        df.to_csv(relax_dir + "/cell.csv")

        # plot rdf
        plot_rdf = kwargs.get('plot_rdf',False)
        if plot_rdf:
            rmax = kwargs.get('rmax',5)

            if any(atoms.cell[i, i] < 3 * rmax for i in range(3)):
                cell = atoms.cell
                repeat = [int(3 * rmax / cell[i, i]) + 1 for i in range(3)]
                atoms = atoms.repeat((repeat[0], repeat[1], repeat[2]))
            if any(atoms_ref.cell[i, i] < 3 * rmax for i in range(3)):
                cell = atoms_ref.cell
                repeat = [int(3 * rmax / cell[i, i]) + 1 for i in range(3)]
                atoms_ref = atoms_ref.repeat((repeat[0], repeat[1], repeat[2]))

            anaref = Analysis(atoms_ref)
            ana = Analysis(atoms)
            rdf_ref = anaref.get_rdf(rmax=rmax, nbins=kwargs.get("nbins", math.ceil(rmax*100)))[0]
            rdf = ana.get_rdf(rmax=rmax, nbins=kwargs.get("nbins", math.ceil(rmax*100)))[0]
            fig, ax = plt.subplots()
            plt.plot(
                np.linspace(0, rmax, kwargs.get("nbins", math.ceil(rmax*100))), rdf, label="MLIP", color="r"
            )
            plt.plot(
                np.linspace(0, rmax, kwargs.get("nbins", math.ceil(rmax*100))),
                rdf_ref,
                label="DFT",
                color="b",
                ls="--",
            )            
            plt.xlabel("r (" + r"$\AA$" + ")")
            plt.ylabel("g(r)")
            plt.legend()
            plt.savefig(os.path.join(relax_dir, f"rdf{rmax}A.png"), bbox_inches="tight")
            plt.close(fig=fig)
        del df
        del images


# class MultiModelOut:
#     # class for calculating uncertainty of multi models
#     def __init__(
#         self,
#         rnd_seed_list,
#         parent_dir,
#         test_data,
#         DFTcalctest_path,
#     ):
#         self.model = []
#         self.rnd_seed_list = rnd_seed_list
#         self.parent_dir = parent_dir
#         self.test_data = test_data
#         self.DFTcalctest_path = DFTcalctest_path
#         for i in range(len(self.rnd_seed_list)):
#             child_dir = self.parent_dir + f"/{rnd_seed_list[i]}/"
#             # # Freeze model
#             # os.chdir(child_dir)
#             # model_name = './graph'
#             # if not os.path.exists(model_name+".pb"):
#             #     subprocess.run(['dp','freeze','-o',model_name+'.pb'])
#             # os.chdir(parent_dir)
#             self.model.append(
#                 ModelOut(
#                     test_data=self.test_data,
#                     calc_path=child_dir + "/frozen_model.pb",
#                     parent_dir=child_dir,
#                     work_path="./",
#                     parity_dir=child_dir + "/calctest/parity/",
#                     eos_dir=child_dir + "/calctest/eos/",
#                     elastic_dir=child_dir + "/calctest/elastic/",
#                     DFTcalctest_path=self.DFTcalctest_path,
#                 )
#             )

#     def MultiEOS(
#         self,
#         cell_list=["primitive_cell"],
#         eos_path="/calctest/EOS/",
#         structure=["boat", "chair"],
#         ranger=[0.95, 1.05],
#         num=25,
#         lattice=["a", "c"],
#         DFTcalctest_path=None,
#         min_volume=False,
#         **kwargs,
#     ):
#         df_list = [dict() for i in range(len(self.rnd_seed_list))]
#         energy = np.empty((len(self.rnd_seed_list), num))

#         if not DFTcalctest_path:
#             DFTcalctest_path = self.DFTcalctest_path

#         # Link QE calctest to the folder
#         os.chdir(self.parent_dir)
#         if not os.path.exists("./QEcalctest"):
#             logging.info(f"Linking to QE calctest at {self.parent_dir}")
#             subprocess.run(["ln", "-s", f"{self.DFTcalctest_path}", "./"])

#         for i, element in enumerate(self.rnd_seed_list):
#             ModelObj = self.model[i]
#             os.chdir(ModelObj.parent_dir)
#             df_list[i] = ModelObj.EOS(
#                 cell_list=cell_list,
#                 eos_path=self.parent_dir + f"/{element}/" + eos_path,
#                 structure=structure,
#                 ranger=ranger,
#                 num=num,
#                 lattice=lattice,
#                 DFTcalctest_path=DFTcalctest_path,
#                 min_volume=False,
#                 **kwargs,
#             )
#             plt.close()
#             # Use a dictionary to store every csv file
#         eos_dict = dict()
#         for stru in structure:
#             for supercell in cell_list:
#                 for lat in lattice:
#                     eos_single = dict()
#                     eos_single["Volume"] = df_list[0][f"{stru}_{supercell}_{lat}"][
#                         "Volume"
#                     ]
#                     for i, element in enumerate(self.rnd_seed_list):
#                         eos_single[f"{element}"] = df_list[i][
#                             f"{stru}_{supercell}_{lat}"
#                         ]["Energy"]
#                         energy[i, :] = df_list[i][f"{stru}_{supercell}_{lat}"]["Energy"]
#                     eos_single["Mean"] = np.mean(energy, axis=0)
#                     eos_single["Std"] = np.std(energy, axis=0)
#                     eos_single["Max"] = np.max(energy, axis=0)
#                     eos_single["Min"] = np.min(energy, axis=0)
#                     eos_single = pd.DataFrame(eos_single)
#                     eos_single.to_csv(
#                         self.parent_dir + eos_path + f"/{stru}_{supercell}_{lat}.csv"
#                     )
#                     eos_dict[f"{stru}_{supercell}_{lat}"] = eos_single

#                     fig, ax = plt.subplots()
#                     eos = EquationOfState(eos_single["Volume"], eos_single["Mean"])
#                     v0, e0, B = eos.fit()
#                     B = B / kJ * 1.0e24
#                     eos.plot(
#                         self.parent_dir
#                         + eos_path
#                         + f"/{stru}_{supercell}_{lat}_{ranger[0]}-{ranger[1]}.png",
#                         ax=ax,
#                         color="r",
#                         label="MLIP mean",
#                         markercolor="r",
#                     )
#                     plt.text(
#                         0.4,
#                         0.8,
#                         f"DeepMD\nB={B:.4f} GPa\nv0={v0:.4f}Å^3\n",
#                         transform=ax.transAxes,
#                         wrap=True,
#                     )
#                     num = kwargs.get("num", 5)
#                     start = kwargs.get("start", 2)
#                     utils.qe_eos(
#                         path=self.parent_dir
#                         + f"/{self.rnd_seed_list[i]}/calctest/EOS/{stru}/qe/{supercell}/{lat}/",
#                         num=num,
#                         start=start,
#                         lattice=lat,
#                         fig=fig,
#                         ax=ax,
#                         txt=True,
#                         plot_path=self.parent_dir
#                         + f"/{stru}_{supercell}_{lat}_{ranger[0]}-{ranger[1]}.png",
#                         color="blue",
#                     )
#                     emin = eos_single["Min"]
#                     emax = eos_single["Max"]
#                     v = eos_single["Volume"]
#                     ax.fill_between(
#                         v, emin, emax, color="r", alpha=0.3, label="Uncertainty"
#                     )
#                     plt.legend()
#                     if not os.path.exists(self.parent_dir + eos_path):
#                         os.makedirs(self.parent_dir + eos_path)
#                     # ax.set_xlim([np.min(v),np.max(v)])
#                     plt.savefig(
#                         self.parent_dir
#                         + eos_path
#                         + f"/{stru}_{supercell}_{lat}_{ranger[0]}-{ranger[1]}.png",
#                         bbox_inches="tight",
#                     )
#                     plt.close()

#     def MultiNEB(
#         self,
#         initial_file,
#         final_file,
#         nimages=5,
#         fmax=0.1,
#         # fixed_size=4,
#         struct_name=None,
#         plot=True,
#         workdir="./calctest/NEB/MLIP/",
#         run_name="boat_lattice",
#         max_steps=400,
#         relax_initial=False,
#         relax_final=False,
#         contraint=False,
#         **kwargs,
#     ):
#         rnd_seed_list = self.rnd_seed_list
#         # initial and final file should be ase-readable format
#         initial = read(initial_file)
#         final = read(final_file)
#         initial_pbc = kwargs.get("initial_pbc", (1, 1, 1))
#         final_pbc = kwargs.get("final_pbc", (1, 1, 1))
#         initial.set_pbc(initial_pbc)
#         final.set_pbc(final_pbc)
#         energy_list = [pd.DataFrame() for i in self.rnd_seed_list]

#         return None

#     def MultiParity(
#         self,
#         test_data=None,
#         parity_dir=None,
#         step=2000,
#         to_csv=True,
#     ):
#         if not test_data:
#             test_data = self.test_data
#         if not parity_dir:
#             parity_dir = self.parent_dir + "/calctest/parity/"
#         if not os.path.exists(parity_dir):
#             os.makedirs(parity_dir)
#         print("start parity")
#         images = read(f"{test_data}@:-1")
#         en = np.empty((len(self.rnd_seed_list), len(images)))
#         fc = []
#         df_e = pd.DataFrame(dict())
#         df_f = pd.DataFrame(dict())

#         df_e["dft"] = np.array(
#             [image.get_potential_energy() / len(image) * 2 for image in images]
#         )
#         df_f["dft"] = np.hstack([image.get_forces().flatten() for image in images])
#         for i in range(len(self.rnd_seed_list)):
#             ModelObj = self.model[i]

#             ModelObj.parity(
#                 parity_dir=None,
#                 parent_dir=self.parent_dir + f"/{self.rnd_seed_list[i]}/",
#                 step=step,
#             )
#             energy = np.empty(len(images))
#             forces = np.array([])
#             for j in range(len(images)):
#                 images[j].calc = ModelObj.calc
#                 energy[j] = images[j].get_potential_energy() / len(images[j]) * 2
#                 forces = np.hstack((forces, images[j].get_forces().flatten()))
#             en[i, :] = energy
#             fc.append(forces)
#             df_e[self.rnd_seed_list[i]] = energy
#             df_f[self.rnd_seed_list[i]] = forces

#         fc = np.array(fc)
#         df_e["mean"] = np.mean(en, axis=0)
#         df_e["max"] = np.max(en, axis=0)
#         df_e["min"] = np.min(en, axis=0)

#         df_f["mean"] = np.mean(fc, axis=0)
#         df_f["max"] = np.max(fc, axis=0)
#         df_f["min"] = np.min(fc, axis=0)

#         if to_csv:
#             df_e.to_csv(parity_dir + "/energy.csv")
#             df_f.to_csv(parity_dir + "/forces.csv")

#         # df_e = pd.read_csv(parity_dir + "/energy.csv")
#         # df_f = pd.read_csv(parity_dir + "/forces.csv")
#         df_e = df_e.sort_values(by=["dft"])
#         df_f = df_f.sort_values(by=["dft"])
#         # if to_csv:
#         #     df_e.to_csv(parity_dir+'/energy.csv')
#         #     df_f.to_csv(parity_dir+'/forces.csv')
#         fig, ax = plt.subplots()
#         xlim = [
#             np.min(df_e["dft"]) - 0.1 * (np.max(df_e["dft"]) - np.min(df_e["dft"])),
#             np.max(df_e["dft"]) + 0.1 * (np.max(df_e["dft"]) - np.min(df_e["dft"])),
#         ]
#         ax.set_xlim(xlim)
#         ax.set_ylim(xlim)
#         plt.plot(xlim, xlim, "--", color="k")
#         rmse = np.sqrt(
#             np.sum(np.square(df_e["dft"] - df_e["mean"])) / len(df_e["mean"])
#         )
#         plt.plot(
#             df_e["dft"],
#             df_e["mean"],
#             "o",
#             color="r",
#             label=f"RMSE={np.round(rmse, 2)*1000}eV/atom",
#         )
#         ax.fill_between(
#             df_e["dft"],
#             df_e["min"],
#             df_e["max"],
#             color="r",
#             alpha=0.4,
#             interpolate=True,
#             label="Uncertainty",
#         )
#         plt.legend()
#         plt.xlabel("DFT Energy per CF (eV)")
#         plt.xlabel("MLIP Energy per CF (eV)")
#         plt.savefig(parity_dir + "/energy.png", bbox_inches="tight")
#         plt.close()

#         fig, ax = plt.subplots()
#         xlim = [np.min(df_f["dft"]) * 1.005, np.max(df_f["dft"]) * 0.995]
#         ax.set_xlim(xlim)
#         ax.set_ylim(xlim)
#         plt.plot(xlim, xlim, "--", color="k")
#         rmse = np.sqrt(
#             np.sum(np.square(df_f["dft"] - df_f["mean"])) / len(df_f["mean"])
#         )
#         plt.plot(
#             df_f["dft"],
#             df_f["mean"],
#             ".",
#             color="r",
#             label=f"RMSE={np.round(rmse, 2)*1000}meV/" + r"$\AA$",
#         )
#         ax.fill_between(
#             df_f["dft"],
#             df_f["min"],
#             df_f["max"],
#             color="r",
#             alpha=0.4,
#             interpolate=True,
#             label="Uncertainty",
#         )
#         plt.legend()
#         plt.xlabel("DFT Energy per CF (eV)")
#         plt.xlabel("MLIP Energy per CF (eV)")
#         plt.savefig(parity_dir + "/forces.png", bbox_inches="tight")
#         plt.close()
