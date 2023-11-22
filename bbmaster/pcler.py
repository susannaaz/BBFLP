import argparse
import healpy as hp
import numpy as np
from bbmaster.utils import PipelineManager, get_pcls


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='pseudo-C_ell calculator')
	parser.add_argument("--globals", type=str,
						help='Path to yaml with global parameters')
	parser.add_argument("--first-sim", type=int, help='Index of first sim')
	parser.add_argument("--num-sims", type=int, help='Number of sims')
	parser.add_argument("--sim-sorter", type=str,
						help='Name of sorting routine')
	parser.add_argument("--output-dir", type=str, help='Output directory')
	parser.add_argument("--sim-type", type=str, help='filtered or input')
	parser.add_argument("--correct-transfer", action='store_true',
						help='Correct for transfer function?')
	parser.add_argument("--filtered-with-toast", action='store_true',
						help='If we filtered with toast, then this should be true')
	parser.add_argument("--beam-fwhm", type=float, help='Beam FWHM in degrees')

	o = parser.parse_args()

	man = PipelineManager(o.globals)
	
	if o.filtered_with_toast:
		filtered_with_toast = True
	else:
		filtered_with_toast = False

	if o.correct_transfer:
		fname = man.get_filename('transfer_function', o.output_dir)
		winv = np.load(fname)['wcal_inv']
	else:
		winv = None
	
	if o.beam_fwhm is not None:
		beam_fwhm = o.beam_fwhm
	else:
		beam_fwhm = None

	sorter = getattr(man, o.sim_sorter)
	b = man.get_nmt_bins()

	sim_names = sorter(o.first_sim, o.num_sims, o.output_dir, which='names')
	file_input_list = sorter(o.first_sim, o.num_sims, o.output_dir,
							 which=o.sim_type)
	file_output_list = sorter(o.first_sim, o.num_sims, o.output_dir,
							  which=o.sim_type+'_Cl')
	mask = hp.ud_grade(hp.read_map(man.fname_mask),
					   nside_out=man.nside)
	for fin, nam, fout in zip(file_input_list, sim_names, file_output_list):
		if isinstance(fin, str):
			fin = [fin]
			nam = [nam]
		get_pcls(man, fin, nam, fout, mask, b, beam_fwhm=beam_fwhm, winv=winv, filtered_with_toast=filtered_with_toast)