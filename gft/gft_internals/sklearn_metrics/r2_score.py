# coding=utf-8
"""r2_score"""

from sklearn.metrics import r2_score,mean_squared_error
import datasets
import numpy as np

_DESCRIPTION = """
See sklearn.metrics.r2_score
"""

_KWARGS_DESCRIPTION = """
To be written
"""

_CITATION = """\
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class R2Score(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Sequence(datasets.Value("float")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("float"),
                    "references": datasets.Value("float"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score"],
        )

    def _compute(self, predictions, references, sample_weight=None, multioutput='uniform_average'):
        score = r2_score(references, predictions, sample_weight=sample_weight, multioutput=multioutput)
        return {"r2_score": float(score) if score.size == 1 else score,
                'rms': mean_squared_error(references, predictions),
                'n': len(predictions),
                'y.mean' : np.mean(references),
                'yhat.mean' : np.mean(predictions)}
    
