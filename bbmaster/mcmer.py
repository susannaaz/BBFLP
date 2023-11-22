import argparse
import numpy as np
import pymaster as nmt
import healpy as hp
import os
from bbmaster.utils import PipelineManager


def mcmer(o):
    man = PipelineManager(o.globals)

    # Number of ells
    nl = 3*man.nside
    # Number of polarisation combinations
    nspec = 4
    
    print(man.fname_mask)

    # Read and degrade mask
    print("Reading mask")
    mask = hp.ud_grade(hp.read_map(man.fname_mask), nside_out=man.nside)
    # Create dummy NaMaster field
    f = nmt.NmtField(mask, None, spin=2)
    # Binning scheme is irrelevant for us, but NaMaster needs one.
    b = man.get_nmt_bins()

    # Alright, compute and reshape coupling matrix.
    # Compute also the binned MCM (this is actually what we need).
    print("Computing MCM")
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f, f, b)
    mcm = np.transpose(w.get_coupling_matrix().reshape([nl, nspec, nl, nspec]),
                       axes=[1, 0, 3, 2])

    # Save to file
    print(o.output_dir)
    os.system(f'mkdir -p {o.output_dir}/mcm') #Added SA
    fname_out = man.get_filename('mcm', o.output_dir)
    print(f"Saving to {fname_out}")
    np.savez(fname_out, mcm=mcm)

    if o.plot:
        import matplotlib.pyplot as plt

        print("Plotting")
        plt.figure()
        plt.imshow(mcm.reshape([nspec*nl, nspec*nl]))
        plt.colorbar()
        fname = os.path.join(man.get_filename("mcm_plots", o.output_dir), #correct SA
                             'mcm.pdf')
        fname_png = os.path.join(man.get_filename("mcm_plots", o.output_dir), #correct SA
                             'mcm.png')
        print(fname_png)
        plt.savefig(fname, bbox_inches='tight')
        plt.savefig(fname_png, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCM calculator')
    parser.add_argument("--globals", type=str,
                        help='Path to yaml with global parameters')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    parser.add_argument("--plot", action='store_true',
                        help='Pass to generate a plot of the MCM.')
    o = parser.parse_args()
    mcmer(o)
