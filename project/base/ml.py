"""Imports most of the useful machine-learning related stuff
from the rest of this folder, for convenience in scripts.

Deliberately does not import the 'fenics' or 'plot' modules.
They are large and separate enough, respectively, to be worth
importing separately.
"""

from .data_gen import (SolutionBase,
                       Peakon,
                       TwoPeakon,
                       sol_on_grid,
                       sol_at_point,
                       gen_one_peakon_on_grid,
                       gen_two_peakon_on_grid,
                       gen_peakons_on_grid,
                       gen_one_peakon_at_point,
                       gen_two_peakon_at_point,
                       gen_peakons_at_point,
                       X_peak,
                       y_peak,
                       BatchData)

from .dnn_from_folder import (dnn_factory_from_model_dir,
                              dnn_factories_from_dir)

from .dnn_from_seq import (Sequential,
                           model_dir_str)

from .ensemble import RegressorAverager

from .evalu import (eval_regressor,
                    eval_regressors)

from .factory import (RegressorFactoryBase,
                      DNNFactory,
                      RegressorFactory)

from .grid import (fine_grid_sep,
                   coarse_grid_sep,
                   num_intervals,
                   fine_grid_fineness,
                   coarse_grid_size,
                   grid,
                   fine_grid,
                   coarse_grid,
                   coarse_grid_center_indices,
                   fine_grid_center_indices)

from .interpolation import (InterpolatorBase,
                            FineGridInterpolator,
                            PointInterpolator,
                            BilinearInterpMixin,
                            PolyInterpMixin,
                            NearestInterpMixin,
                            FineGridBilinearInterp,
                            FineGridPolyInterp,
                            PointBilinearInterp,
                            PointPolyInterp,
                            Perfect)

from .processor import (ProcessorBase,
                        IdentityProcessor,
                        ScaleOverall,
                        NormalisationOverall,
                        ScaleCenter,
                        AtPointMixin,
                        ScaleCenterAtPoint,
                        ProcessorSavingHook)
