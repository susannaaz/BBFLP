# BBFLP - the First Light Pipeline for BB (Tutorials)
----------------------------------------------------

Overview for generating and pre-processing Simons Observatory (SO) SAT BB data products and manipulating these products, using processes that have been developed by numerous people within SO (it is to be thought of as a collection of useful tools, currently intended for personal use). 

The Jupyter notebook tutorials are focused on using toast/sotodlib for simulating and pre-processing data, including filter-bin mapmaking, and BBMASTER (a.k.a. SOOPERCOOL) for estimating unbiased power spectra.


## Table of contents

 --- 
> [Section 0 - Introduction](Sec1_Introduction.ipynb): Overview of contents.

> [Section 1 - Setting up on NERSC](Sec1_Setup_NERSC.ipynb): Guide on how to run these notebooks on NERSC.

> [Section 2 - Sky maps](Sec2_Sky_maps.ipynb): How to generate simple sky maps, including CMB and/or foregrounds using [PySM](https://github.com/galsci/pysm). Ongoing: integrate with existing tools (e.g. [susannaaz/BBSims](https://github.com/susannaaz/BBSims) or [simonsobs/BBSims](https://github.com/simonsobs/BBSims)) based on [PySM](https://github.com/galsci/pysm) and [CAMB](https://github.com/cmbant/CAMB).

> [Section 3 - Schedule](Sec3_Make_and_Analyze_schedule.ipynb): Guide to creating a schedule using toast, reading a schedule, and splitting it into singular observations. 

> [Section 4 - TOD simulation](Sec4_Simulate_TOD_breakdown.ipynb): Introduction to [toast3](https://github.com/hpc4cmb/toast/tree/toast3) and [sotodlib](https://github.com/simonsobs/sotodlib). Here we explain how to generate Time-Ordered-Data (TOD) simulations. It is explained how to create a focal plane, and from a given schedule create a telescope, include boresight pointing information, weights, add the sky signal from some input map (that can be generated as shown in [Section 2](Sec2_Sky_maps.ipynb), and noise.  This is saved as an AxisManager in hdf5 format, which is compatible with the latest [PWG SAT simulations](https://github.com/simonsobs/pwg-scripts/tree/master/pwg-tds/pipe-s0002/v6), and can be easily coverted into books, to resemble the format of actual observations. Section 3 and Section 4 reproduce what is created with toast as in [Simulation_tod_TOAST](Simulate_tod_TOAST.ipynb), but allows us to visualize the intermediate steps and modify them without running the full simulation.

> [Section 5 - Pre-process the data and make maps](Sec5_Preprocess_TOD_Make_maps.ipynb): Demonstration of methods to pre-process the TOD using different operations and filters applied to the data, and create maps using filter-bin map-making as implemented in sotodlib. This emulates how the real data will be processed. A demonstration of methods to convert maps between CAR and HEALPix format is included.

> [Section 6 - Master pipeline](Sec6_Master_pipeline.ipynb): Demonstration of integration with the Master pipeline for power spectrum estimation, using the tools implemented in [BBMaster](https://github.com/simonsobs/BBMASTER).

> [Section 7 - Estimate r](Sec7_Estimate_r.ipynb): This is a ``dirty'' way of estimating of the tensor-to-scalar ratio, which assumes a single frequency observation of a CMB-only sky, *not* to be used for real observations. Eventually, we want to repeat the pipeline demonstrated in the previous sections to integrate foregrounds, and this section will be substituted with the integration with component separation pipelines, e.g. BPower.