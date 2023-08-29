#force_fit
A force fitting software for radio images, currently focussed on 
ASKAP and VLASS images. 

## Intention behind this code
Source fitting packages like Selavy/Aegean/Pydbsf are routinely 
used for finding radio sources in survey images. Given a noise
threshold (usually 5-sigma), these routines will produce source
catalogs. But if we want to look at faint sources (say 3-sigma)
or do a forced fitting at a given sky position or handful of 
positions, running these routines will be an overkill. Currently
there exists a force fit routine for ASKAP images, however its
limitation is that it can not search around in the positional 
space to look for source, meaning if there a source that is 
offset from a given position, the existing code can not fit for
the position and flux simutaneously. This will result in lower 
limits at best (instead of intended upper limits). 

This package is intended to bridge the gap between these two
approaches. If we have a handful of sources (on an order of 
a few hundereds), this code can look around the given positions
to find for sources. It will look for point sources in the images.

## Limitations
This code is not intended to be used a scaled Selvay/Pydbsf. It 
is not extremely optimized between speed and source searching.
It can robustly fit for sources but is not scalable to all the
survey images. And it can currently it can fit only for point 
sources.

# Usage
It can be used as follows:

