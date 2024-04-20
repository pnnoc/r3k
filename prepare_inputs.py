import argparse
import re
import numpy as np
import uproot as ur
import awkward as ak
import multiprocessing as mp
from utils import preprocess_files


def preprocess_inputs(runFiles, ipart, args, branch_dict):
    ax_flat = None if args.flat else 1
    if 'train' in args.mode:
        args.useLowQ = True
        if 'data' in args.channel:
            args.useBsideBands = True
    elif args.mode=='measure':
        args.useLowQ = False
        args.useBsideBands = False

    tree_values={}
    ncands_branch = 'n'+branch_dict['candidate']
    needed_branches = list({ncands_branch} | branch_dict['cand_branches'].keys() | branch_dict['scalar_branches'].keys())
    for i, tree in enumerate(ur.iterate([runFile+':Events' for runFile in runFiles],needed_branches,cut=args.split,namedecode='utf-8',library='np')):
        presel_mask = np.full(ak.flatten(tree[branch_dict['candidate']+'_fit_mass'],ax_flat).to_numpy().shape, True)
        # we want to rearrange leptons to make sure that they are properly sorted (pt or type). so the output leptonX will have two contributions from input leptonX and Y
        outl1_inl1_mask = np.copy(presel_mask)
        outl1_inl2_mask = np.copy(presel_mask)

        # Deal with scalars
        entries_per_evt = tree[ncands_branch]
        scalars = {br : np.repeat(tree[br],1 if args.flat else entries_per_evt) for br in branch_dict['scalar_branches'].keys()}

        sortby = 'leppt'
        if sortby=='eltype':
             id1 = ak.flatten(tree[branch_dict['candidate']+'_l1_isPF'],ax_flat)
             id2 = ak.flatten(tree[branch_dict['candidate']+'_l2_isPF'],ax_flat)
             outl1_inl1_mask = np.where(id1==1,1,0)
             outl1_inl2_mask = np.where(id2==1,1,0)
        else:
             pt1 = ak.flatten(tree[branch_dict['candidate']+'_fit_l1_pt'],ax_flat)
             pt2 = ak.flatten(tree[branch_dict['candidate']+'_fit_l2_pt'],ax_flat)
             outl1_inl1_mask = np.where(pt1>pt2,1,0) + np.where(pt1==pt2,1,0)
             outl1_inl2_mask = np.where(pt2>pt1,1,0)

        outl2_inl1_mask = 1 - outl1_inl1_mask
        outl2_inl2_mask = 1 - outl1_inl2_mask

        #remove infs from data
        copied_branches = {}
        inf_mask = np.full(len(outl1_inl1_mask),True)
        nan_mask = np.full(len(outl1_inl1_mask),True)
        for branch in needed_branches:
            copied_branches[branch] = ak.flatten(tree[branch],ax_flat)
            infs = np.argwhere(np.isinf(ak.flatten(tree[branch],ax_flat)))
            nans = np.argwhere(np.isnan(ak.flatten(tree[branch],ax_flat)))
            for idx in infs:
                inf_mask[idx] = False
                np.asarray(copied_branches[branch])[idx] = 0
            for idx in nans:
                nan_mask[idx] = False
                np.asarray(copied_branches[branch])[idx] = 0

        presel_mask = np.full(len(copied_branches[branch_dict['candidate']+'_fit_mass']),True)
        for cut in branch_dict['presel']:
            pattern = '|'.join(map(re.escape, ['>', '<', '>=', '<=']))
            br, op, val = [part.strip() for part in re.split(f"({pattern})", cut) if part]
            mask = eval(f'np.array((copied_branches["{br}"] {op} {val}))')
            presel_mask *= mask

        presel_mask *= inf_mask
        presel_mask *= nan_mask
        mB_branch = copied_branches[branch_dict['candidate']+'_fit_mass']
        mll_branch = copied_branches[branch_dict['candidate']+'_mll_fullfit']
        if args.useBsideBands:
            sidebands = np.array(branch_dict['candidate_mass_sidebands'])
            if sidebands.ndim==1:
                presel_mask *= np.array((mB_branch>sidebands[0]) * (mB_branch<sidebands[1]))
            else:
                assert sidebands.ndim==2
                presel_mask *= np.logical_or.reduce([np.logical_and(mB_branch>=start, mB_branch<=end) for start, end in sidebands])
        else:
            presel_mask *= np.array((mB_branch>branch_dict['candidate_mass_range'][0]) * (mB_branch<branch_dict['candidate_mass_range'][1]))
        if args.useLowQ:
            presel_mask *= np.array((mll_branch>branch_dict['lowq2_region'][0]) * (mll_branch<branch_dict['lowq2_region'][1]))
        if args.useHighQ:
            presel_mask *= np.aarray((mll_branch>branch_dict['highq2_region'][0]) * (mll_branch<branch_dict['highq2_region'][1]))

        #eta cuts k, e1,e2
        k_eta_branch = copied_branches[branch_dict['candidate']+'_fit_k_eta']
        l1_eta_branch = copied_branches[branch_dict['candidate']+'_fit_l1_eta']
        l2_eta_branch = copied_branches[branch_dict['candidate']+'_fit_l2_eta']
        presel_mask *= np.array((k_eta_branch<2.4) * (k_eta_branch>-2.4))
        presel_mask *= np.array((l2_eta_branch<2.4) * (l2_eta_branch>-2.4) * (l1_eta_branch<2.4) * (l1_eta_branch>-2.4))

        outleps={}
        for br1, br2 in branch_dict['leppairs_branches'].items():
            #if exists take it from cleaned for e1
            if br1 in copied_branches.keys():
                outleps[br1] = copied_branches[br1] * outl1_inl1_mask + copied_branches[br2] * outl1_inl2_mask
            else:
                outleps[br1] = ak.flatten(tree[br1],ax_flat) * outl1_inl1_mask + ak.flatten(tree[br2],ax_flat) * outl1_inl2_mask
            #alternative id from tree -- pray not to have inf
            if 'Id' in br1 and sortby=='eltype':
                branchId_change = [] # Add if switching branch ID
                outleps[br2] = ak.flatten(tree[branchId_change[0]],ax_flat) * outl2_inl1_mask + ak.flatten(tree[branchId_change[1]],ax_flat) * outl2_inl2_mask
            else:
                #for e2
                if br2 in copied_branches.keys():
                    outleps[br2] = copied_branches[br1] * outl2_inl1_mask + copied_branches[br2] * outl2_inl2_mask
                else:
                    outleps[br2] = ak.flatten(tree[br1],ax_flat) * outl2_inl1_mask + ak.flatten(tree[br2],ax_flat) * outl2_inl2_mask

        for cut in branch_dict['leppairs_presel']:
            pattern = '|'.join(map(re.escape, ['>', '<', '>=', '<=']))
            br, op, val = [part.strip() for part in re.split(f"({pattern})", cut) if part]
            mask = eval(f'np.array((outleps["{br}"] {op} {val}))')
            presel_mask *= mask

        output_branches = {**branch_dict['cand_branches'], **branch_dict['scalar_branches']}
        for br, br_name in output_branches.items():
            #check if it is a lepton
            if br in outleps.keys():
                selected_evts = outleps[br][presel_mask]
            #check if it a scalar
            elif br in scalars.keys():
                selected_evts = scalars[br][presel_mask]
            else:
                if br in copied_branches.keys():
                     selected_evts = ak.flatten(tree[br],ax_flat)[presel_mask]
                else:
                     selected_evts = copied_branches[br][presel_mask]

            if br_name not in tree_values.keys():
                 tree_values.update({br_name:selected_evts})
            else:
                 tree_values[br_name]=np.concatenate((tree_values[br_name], selected_evts))

    outname = 'measurement' if args.mode=='measure' else 'train'
    tags = [
        (args.channel=='data' and args.mode=='train', '_bkg'),
        (args.channel=='rare' and args.mode=='train', '_sig'),
        (True, f'_{args.channel}'),
        (args.useBsideBands, '_sideBands'),
        (args.useLowQ, '_lowQ'),
        (args.useHighQ, '_highQ'),
        (args.total>0, f'_maxFiles_{str(args.total)}'),
        (args.label, f'_{args.label}'),
        (ipart, f'_part{ipart}'),
    ]
    outname += ''.join(tag for cond, tag in tags if cond)

    with ur.recreate(args.outpath+'/'+outname+'.root') as f:
        f['mytree'] = tree_values


def mp_worker(files,ipart,args,branch_dict):
    if args.mode=='split':
        args.mode='train'
        args.split='event%2==0'
        preprocess_inputs(files,ipart,args,branch_dict)
        args.mode='measure'
        args.split='event%2!=0'
        preprocess_inputs(files,ipart,args,branch_dict)
    else:
        preprocess_inputs(files,ipart,args,branch_dict)

    print(f'Part {ipart} Finished')


def main(args):
    # parameters
    col = 'BToKEE'
    args.nparts = args.nparts if args.nparts > 0 else mp.cpu_count()
    # args.total = args.total if args.total > 0 else
    args.sortby = 'leppt' #options: leppt, eltype. sort leptons by pt or electron type
    args.useLowQ = False
    args.useHighQ = False
    args.useBsideBands = False

    # Define type of events
    if args.channel:
        pass
    elif 'data' in args.inpath.lower():
        args.channel = 'data'
    elif 'kee' in args.inpath.lower():
        args.channel = 'rare'
    elif 'jpsi' in args.inpath.lower():
        args.channel = 'jpsi'
    elif 'psi2s' in args.inpath.lower():
        args.channel = 'psi2s'
    else:
        args.channel = 'data'

    # input file parameters
    jobFiles = preprocess_files(args.inpath, args.nparts, args.total)

    if 'KEE' in col:
        branch_dict = {
            'candidate'                : col,
            'candidate_mass_range'     : (4.5,6.),
            'candidate_mass_sidebands' : ((4.8,5.),(5.4,5.6)),
            'lowq2_region'             : (1.05,2.45),
            'highq2_region'            : (3.85,6.),
            'cand_branches'            : {
                    col+'_mll_fullfit'     : 'Mll',
                    col+'_fit_pt'          : 'Bpt',
                    col+'_fit_mass'        : 'Bmass',
                    col+'_fit_cos2D'       : 'Bcos',
                    col+'_svprob'          : 'Bprob',
                    col+'_fit_massErr'     : 'BmassErr',
                    col+'_b_iso04'         : 'Biso',
                    col+'_l_xy_sig'        : 'BsLxy',
                    col+'_fit_l1_pt'       : 'L1pt',
                    col+'_fit_l1_eta'      : 'L1eta',
                    col+'_l1_iso04'        : 'L1iso',
                    col+'_l1_PFMvaID_retrained'      : 'L1id',
                    col+'_fit_l2_pt'       : 'L2pt',
                    col+'_fit_l2_eta'      : 'L2eta',
                    col+'_l2_iso04'        : 'L2iso',
                    col+'_l2_PFMvaID_retrained'      : 'L2id',
                    col+'_fit_k_pt'        : 'Kpt',
                    col+'_k_iso04'         : 'Kiso',
                    col+'_fit_k_eta'       : 'Keta',
                    col+'_lKDz'            : 'LKdz',
                    col+'_lKDr'            : 'LKdr',
                    col+'_l1l2Dr'          : 'L1L2dr',
                    col+'_k_svip3d'        : 'Kip3d',
                    col+'_k_svip3d_err'    : 'Kip3dErr',
                    col+'_l1_iso04_dca'    : 'L1isoDca',
                    col+'_l2_iso04_dca'    : 'L2isoDca',
                    col+'_k_iso04_dca'     : 'KisoDca',
                    col+'_b_iso04_dca'     : 'BisoDca',
                    col+'_k_dca_sig'       : 'KsDca',
                    col+'_kl_massKPi'      : 'KLmassD0',
                    col+'_p_assymetry'     : 'Passymetry',
            },
            'leppairs_branches' : {
                    col+'_fit_l1_pt'   : col+'_fit_l2_pt',
                    col+'_fit_l1_eta'  : col+'_fit_l2_eta',
                    col+'_l1_PFMvaID_retrained'  : col+'_l2_PFMvaID_retrained',
                    col+'_l1_iso04'    : col+'_l2_iso04',
            },
            'scalar_branches' : {
                    'PV_npvs'    : 'Npv',
                    'event'      : 'idx',
                    'Presel_BDT' : 'presel_bdt',
            },
            'presel' : {
                    f'{col+"_svprob"} > 0.0001',
                    # f'{col+"_fit_cos2D"}   > 0.9',
                    # f'{col+"_fit_pt"}      > 0.0',
                    # f'{col+"_l_xy_sig"}    > 2.0',
                    f'{col+"_fit_k_pt"}    > 0.5',
                    f'{col+"_mll_fullfit"} > 0.0',
                    'Presel_BDT > -3.4',
            },
            'leppairs_presel' : {
                    f'{col+"_fit_l1_pt"}  > 1.0',
                    f'{col+"_fit_l2_pt"}  > 1.0',
                    # f'{col+"_l1_PFMvaID_retrained"} > -1.5',
                    # f'{col+"_l2_PFMvaID_retrained"} > -3.0',
            },
        }

    elif 'KMuMu' in col:
        pass
    else:
        raise KeyError('pick allowed column name')

    if 'data' not in args.channel:
        branch_dict['scalar_branches'].update({'trig_wgt' : 'trig_wgt'})

    if args.nparts>1:
        print(f'Distributing {args.total} Files to {args.nparts} workers...')

        procs = []
        for i, ifiles in enumerate(jobFiles):
            print(f'Submitting Part {str(i+1)}')
            proc = mp.Process(target=mp_worker, args=(ifiles,i+1,args,branch_dict))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

    else:
        runFiles = jobFiles

        if args.mode=='split':
            args.mode='train'
            args.split='event%2==0'
            preprocess_inputs(runFiles,None,args,branch_dict)
            args.mode='measure'
            args.split='event%2!=0'
            preprocess_inputs(runFiles,None,args,branch_dict)

        else: preprocess_inputs(runFiles,None,args,branch_dict)


if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', dest='mode', type=str, default='measure')
    parser.add_argument('-j', '--nparts', dest='nparts', type=int, default=1)
    parser.add_argument('-N', '--max-files', dest='total', type=int, default=-1)
    parser.add_argument('-s', '--split', dest='split', action='store_false')
    parser.add_argument('-i', '--inpath', dest='inpath', type=str, default='/eos/cms/store/group/phys_bphys/DiElectronX/jodedra/FullRunThrough_21_09_23/BDTscoreoutput/outputwithnocuts_08_11_23')
    parser.add_argument('-o', '--outpath', dest='outpath', type=str, default='.')
    parser.add_argument('-l', '--label', dest='label', type=str, default='')
    parser.add_argument('-f', '--flat', dest='flat', action='store_true')
    parser.add_argument('-c', '--channel', dest='channel', type=str, default=None)
    args=parser.parse_args()

    main(args)
