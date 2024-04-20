import os
import sys
import logging
import importlib.util
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import uproot as ur
from xgboost import Booster
from joblib import dump,load
from glob import glob
from pathlib import Path
from sklearn.metrics import auc, RocCurveDisplay
from logging import DEBUG, INFO, WARNING, ERROR

BACKEND = 'np'


class R3KLogger():
    def __init__(self, filepath, verbose=True):        
        self.filepath = filepath
        self.verbose = verbose
        
        self.base_logger = logging.getLogger('base_logger')
        self.base_logger.setLevel(logging.INFO)
        self.stdout_logger = logging.getLogger('stdout_logger')
        self.stdout_logger.setLevel(logging.INFO)
        self.fout_logger = logging.getLogger('fout_logger')
        self.fout_logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(levelname)s | %(asctime)s | %(message)s')

        self.fh = logging.FileHandler(self.filepath, mode='w')
        self.fh.setFormatter(self.formatter)
        self.fh.setLevel(logging.INFO)

        self.sh = logging.StreamHandler(sys.stdout)
        self.sh.setFormatter(self.formatter)
        self.sh.setLevel(logging.INFO)

        self.base_logger.addHandler(self.sh)
        self.stdout_logger.addHandler(self.sh)
        self.base_logger.addHandler(self.fh)
        self.fout_logger.addHandler(self.fh)


    def log(self, string, just_print=False, just_write=False, level=INFO):
        if just_print:
            if self.verbose:
                self.stdout_logger.log(level, string)
        elif just_write:
            self.fout_logger.log(level, string)
        else:
            if self.verbose:
                self.base_logger.log(level, string)
            else:
                self.fout_logger.log(level, string)


class ROCPlotterKFold():
    def __init__(self, kf):
        self.kf = kf
        self.ifold = 0
        self.tprs = []
        self.aucs = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

    def add_fold(self, model, X, y):
        self.ifold += 1
        _viz = RocCurveDisplay.from_estimator(
            model,
            X,
            y,
            name=f'ROC fold {self.ifold}',
            alpha=0.3,
            lw=1,
            ax=self.ax,
            plot_chance_level=(self.ifold ==self.kf.get_n_splits() - 1),
        )

        _interp_tpr = np.interp(self.mean_fpr, _viz.fpr, _viz.tpr)
        _interp_tpr[0] = 0.0
        self.tprs.append(_interp_tpr)
        self.aucs.append(_viz.roc_auc)

    def save(self, path, show=False, zoom=False):
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        self.ax.plot(
            self.mean_fpr,
            mean_tpr,
            color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        self.ax.fill_between(
            self.mean_fpr,
            tprs_lower,
            tprs_upper,
            color='grey',
            alpha=0.2,
            label=r'$\pm$ 1 std. dev.',
        )

        self.ax.set(
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            title='Mean ROC curve with variability\n(Positive label "Signal")',
        )
        self.ax.legend(loc='lower right')
    
        if zoom:
            self.ax.set_xlim([0.,1.])
            # self.ax.set_xscale('log')
            self.ax.set_ylim([.9,1.05])
            # self.ax.set_xscale('log')

        if show:
            plt.show(block=True)

        plt.savefig(path)


def read_bdt_arrays(file, tree, features, weights_branch=None, preselection=None, cutvar_branches=('Bmass', 'Mll'), n_evts=None):
    all_branches = list(set(features) | set(cutvar_branches))
    if weights_branch:
        all_branches += [weights_branch]

    with ur.open(file) as f:
        all_arrays = f[tree].arrays(all_branches, cut=preselection, entry_stop=n_evts, library=BACKEND)

    features_array = np.stack([all_arrays[k] for k in features]).T
    cutvars_dict  = {k:all_arrays[k] for k in cutvar_branches}
    weights_array  = all_arrays[weights_branch] if weights_branch else np.ones(features_array.shape[0])

    return features_array, cutvars_dict, weights_array


def save_bdt_arrays(input_file, input_tree, output_file, output_tree, output_branch_names, score_branch, scores, idxs=None, preselection=None, n_evts=None):    
    with ur.open(input_file) as f_in:
        output_branches = f_in[input_tree].arrays(output_branch_names, cut=preselection, entry_stop=n_evts, library=BACKEND)

        if idxs is not None:
            for br in output_branches.values():
                br = br[idxs]

        output_branches[score_branch] = scores
        output_branches['trigger_OR'] = np.ones_like(scores)

        with ur.recreate(output_file, compression=ur.LZMA(9)) as f_out:
            f_out[output_tree] = output_branches


def load_external_model(filepath, debug=False, model_name='model'):
    spec = importlib.util.spec_from_file_location('tmp_module', filepath)
    source_module = importlib.util.module_from_spec(spec)
    sys.modules['tmp_module'] = source_module
    spec.loader.exec_module(source_module)
    model = getattr(source_module,model_name)

    assert callable(getattr(model, 'fit')) and callable(getattr(model, 'predict_proba'))

    if debug:
        model.set_params(**{'n_estimators' : 5})

    return model

def get_branches(output_params, branch_names):
    output_branches = []
    for key in branch_names:
        if output_params.output_branches[key] is not None:
            output_branches.extend(output_params.output_branches[key])
    
    return output_branches


def save_model(output_name, model, args, formats, logger):
    if '.pkl' in formats:
        name = os.path.join(args.outdir, output_name+'.pkl')
        dump(model, name)
        if logger:
            logger.log(f'Saving Model {name}')
    if '.text' in formats:
        name = os.path.join(args.outdir, output_name+'.text')
        booster = model.get_booster()
        booster.dump_model(name, dump_format='text')
        if logger:
            logger.log(f'Saving Model {name}')
    if '.json' in formats:
        name = os.path.join(args.outdir, output_name+'.json')
        model.save_model(name)
        if logger:
            logger.log(f'Saving Model {name}')
    if '.txt' in formats:
        name = os.path.join(args.outdir, output_name+'.txt')
        model.save_model(name)
        if logger:
            logger.log(f'Saving Model {name}')


def load_bdt(args):
    args.filepath = args.model if args.format in args.model else args.model+args.format
    assert os.path.exists(args.filepath)

    if ('pkl' in args.format) or ('pickle' in args.format):
        return load(args.filepath)
    else:
        bdt = Booster()
        bdt.load_model(args.filepath)
        return bdt 


def preprocess_files(input_files, nparts, total):
    filelist = [input_files] if input_files.endswith('.root') else glob(input_files+'/**/*.root',recursive=True)[:total]
    if nparts==1:
        outfiles = filelist
    else:
        outfiles = np.array_split(np.array(filelist), nparts if nparts!=-1 else mp.cpu_count())

    if not outfiles: 
        raise ValueError('Invalid input path/file')
    return outfiles


def check_rm_files(files=[]):
    for fl in files:
        if os.path.isfile(fl):
            os.system('rm '+fl)


def edit_filename(path, prefix='', suffix=''):
    path = Path(path)
    path = path.with_stem('_'.join(filter(None, [prefix, str(path.stem), suffix])))
    return path