# CloudFlex

CloudFlex is an open-source Python package for modeling the observational
signatures associated with the small-scale structure of the circumgalactic medium.
It populates cool gas structures in the CGM as a complex of cloudlets using
a Monte Carlo method.  Various parameters can be set to describe the 
structure of the cloudlet complexes, including cloudlet mass, density, velocity,
and size.  Functionality exists for generating the observational signatures
of sightlines piercing these cloudlet complexes, borrowing heavily from the 
Trident code.

For basic use of the code, you can manually set your parameters in `make_clouds.py`,
and then run it to generate a cloudlet complex as an HDF5 file:

```
python make_clouds.py
```

Following this, you can send a large number of random sightlines through this cloudlet
complex:

```
python make_rays.py clouds.h5
```

To systematically vary one parameter over several different values and compare the
observational results against each other, try running:

```
python make_all.py
```

Our method paper for CloudFlex will be on the arXiv shortly.

--Cameron Hummels, Kate Rubin, Evan Schneider, and Drummond Fielding
