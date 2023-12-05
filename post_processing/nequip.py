from MLIP_processing.post_processing.model_test import ModelOut
from calctest.nequip import NequIPTest
from MLIP_processing.post_processing.multi_model_test import MultiModelOut
import os

class NequipOut(ModelOut,NequIPTest):
    def __init__(
        self,
        test_data,
        calc_path,
        parent_dir,
        work_path="./",
        parity_dir="./calctest/parity/",
        eos_dir="./calctest/eos/",
        elastic_dir="./calctest/elastic/",
        DFTcalctest_path=None,
        device = 'cuda',
        **kwargs,
    ):
        self.calc_params = {'calc_path':calc_path}
        super(NequipOut,self).__init__(
            calc_params = self.calc_params,
            test_data = test_data,
            calc_path = calc_path,
            parent_dir = parent_dir,
            work_path = work_path,
            parity_dir = parity_dir,
            eos_dir = eos_dir,
            elastic_dir = elastic_dir,
            DFTcalctest_path = DFTcalctest_path,
            calc_name = kwargs.get('calc_name','Nequip')
            )
        self.device = device

class MultiNequipOut(MultiModelOut):
    def __init__(
        self,
        model_ensemble_paths,
        parent_dir,
        test_data,
        DFTcalctest_path,
        device='cuda',
        **kwargs,
    ):
        models = []
        for i, model_path in enumerate(model_ensemble_paths):
            model_parent = os.path.join(parent_dir,model_path.removesuffix('deployed_model.pth'))
            models.append(
                NequipOut(
                    test_data=test_data,
                    calc_path= os.path.join(parent_dir,model_path),
                    parent_dir = model_parent,
                    work_path="./",
                    parity_dir="./calctest/parity/",
                    eos_dir="./calctest/eos/",
                    elastic_dir="./calctest/elastic/",
                    DFTcalctest_path=None,
                    device = 'cuda',
                    calc_name = kwargs.get('calc_name','Nequip')
                )
            )
        super().__init__(
            models = models,
            model_ensemble_paths = [os.path.join(parent_dir,model_path) for model_path in model_ensemble_paths],
            parent_dir = parent_dir,
            test_data = test_data,
            DFTcalctest_path = DFTcalctest_path,
        )