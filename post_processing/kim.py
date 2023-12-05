from calctest.kim import KIMTest
from MLIP_processing.post_processing.model_test import ModelOut
from MLIP_processing.post_processing.multi_model_test import MultiModelOut
import os

class KIMout(ModelOut,KIMTest):
    def __init__(
        self,
        test_data,
        parent_dir,
        calc_params=None,
        **kwargs,
    ):
        super().__init__(
            test_data = test_data,
            calc_params = calc_params,
            work_path = kwargs.get('work_path',"./"),
            parity_dir = kwargs.get('parity_dir',"./calctest/parity/"),
            parent_dir = parent_dir,
            eos_dir = kwargs.get('eos_dir',"./calctest/eos/"),
            elastic_dir = kwargs.get('elastic_dir',"./calctest/elastic/"),
            DFTcalctest_path = kwargs.get('DFTcalctest_path',None),
            calc_name = kwargs.get('calc_name','KIM'),
        )


# class MultiKIMout(MultiModelOut):
#     # class for calculating uncertainty of multi models
#     def __init__(
#         self,
#         model_list,
#         parent_dir
#         model_parent_dir_list,
#         test_data,
#         DFTcalctest_path,
#         **kwargs,
#     ):
#         models = []
#         for i,model in enumerate(model_list):
#             models.append(
#                 KIMout(
#                     test_data = test_data,
#                     calc_path = None,
#                     parent_dir = model_parent_dir_list[i],
#                     **kwargs,
#                 )
#             )
#         super().__init__(
#             models = models,
#             model_ensemble_paths = model_parent_dir_list,
#             parent_dir = parent_dir,
#             test_data = test_data,
#             DFTcalctest_path = DFTcalctest_path,
#             )    