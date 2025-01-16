# CloudFlex

CloudFlex is an open-source Python package for modeling the observational
signatures associated with the small-scale structure of the circumgalactic medium.
It populates cool gas structures in the CGM as a complex of cloudlets using
a Monte Carlo method.  Various parameters can be set to describe the
structure of the cloudlet complexes, including cloudlet mass, density, velocity,
and size.  Functionality exists for generating the observational signatures
of sightlines piercing these cloudlet complexes, borrowing heavily from the
[Trident code](https://github.com/trident-project/trident).

## Installation

CloudFlex is under active development, with a formal release of the code forthcoming.

Until that time, users who want to run CloudFlex are encouraged to download/clone
this repository, ensure they have the dependencies using pip or conda, and download
the Trident ion balance file.

### Dependencies

 * HDF5
 * h5py
 * numpy
 * scipy
 * astropy
 * matplotlib
 * tqdm
 * unyt
 * cmyt
 * cmocean

### Trident Ion Balance File

CloudFlex requires the ion balance file from the
[Trident code](https://github.com/trident-project/trident) in order to calculate
ionization fractions based on varying thermodynamic properties of the gas.  Existing
Trident installations already have this file, but new users can download it from
[here](http://trident-project.org/data/ion_table/hm2012_hr.h5.gz), gunzip it, and
place it in either the `~/.trident` directory or in the CloudFlex source directory.

## Usage

For basic use of CloudFlex, you can manually set your model parameters in
`make_clouds.py`, and then run it to generate a cloudlet complex as an HDF5 file:

```
python make_clouds.py
```

Following this, you can send a large number of random sightlines through this cloudlet
complex using the wrapper file `make_rays.py`:

```
python make_rays.py clouds.h5
```

To systematically vary one parameter over several different values and compare the
observational results against each other, try running:

```
python make_all.py
```

## Method Paper

Our method paper for CloudFlex gives a full description of the methods used in the code.
If you use CloudFlex in your work, please consider citing this paper:

Hummels & Rubin et al. (2024): [https://ui.adsabs.harvard.edu/abs/2024ApJ...972..148H/abstract](https://ui.adsabs.harvard.edu/abs/2024ApJ...972..148H/abstract)

--Cameron Hummels, Kate Rubin, Evan Schneider, and Drummond Fielding
