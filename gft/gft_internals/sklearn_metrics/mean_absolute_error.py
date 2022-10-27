# coding=utf-8
"""mean_absolute_error"""

from sklearn.metrics import mean_absolute_error
import datasets


_DESCRIPTION = """
See sklearn.metrics.mean_absolute_error
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
class MeanAbsoluteError(datasets.Metric):
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
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html"],
        )

    def _compute(self, predictions, references, sample_weight=None, multioutput='uniform_average'):
        score = mean_absolute_error(references, predictions, sample_weight=sample_weight, multioutput=multioutput)
        return {"mean_absolute_error": float(score) if score.size == 1 else score}
