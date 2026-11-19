# Third-party notices

The combined Python port is distributed under **GPL-3.0-or-later**. The original
orthoDr and dr components remain available under their original
**GPL-2.0-or-later** terms. This port exercises the later-version option for
combination with Apache-2.0 Armadillo headers. The full GPL version 3 is in
`LICENSE`; component notices and license texts are in `LICENSES/`.

## orthoDr

The numerical kernels and translated R interfaces derive from orthoDr 2.0.0,
source checkpoint `a9588d8dd73c39188d1c46f371763addb10bee9a`, a coordinated
development revision based on commit `d99bfa869ab211f974a7a045e8fa6ede23639a48`
of <https://github.com/teazrq/orthoDr>.

The pinned DESCRIPTION credits authors Ruilin Zhao, Wenzhuo Zhou, and Ruoqing
Zhu, contributors Jiyang Zhang and Peng Xu, and copyright holder Ruoqing Zhu.
The upstream package license is GPL (>= 2).

`src/cpp/` retains the upstream notices. The rewritten pseudo-direct solver
uses a short source header; its original license and warranty notice is
preserved in `LICENSES/orthoDr-source-notice.txt`. `tools/vendor_core.py` records original
and generated SHA256 hashes in `src/cpp/provenance.json`. Changes made for this
port on 2026-11-16 replace Rcpp data conversion, exceptions, output and interrupt
handling with a Python bridge. The 2026-11-17 revision adapts the corrected
pseudo-direct native solver from the matching R source. A fingerprinted update
from the R development source revises regression moments and optimization,
survival line searches, and the dose fitting and prediction kernels. The
callback optimizer and high-level R routines are translated to
Python, and the Python API adds explicit input validation and fitted objects.

## dr

The SIR, SAVE, residual PHD, slice construction, whitening, and grouped SAVE
implementations in `orthodr/initializers.py` are translated from the R package
dr 3.0.11, authored by Sanford Weisberg. The upstream license is GPL (>= 2).
Source: <https://CRAN.R-project.org/package=dr>. No R installation is needed at
runtime. The coordinated pSAVE revision transports local SAVE moments into
common coordinates before pooling and returns directions in the original
predictor units. `LICENSES/GPL-2.0.txt` preserves the original license text.

## Armadillo

The extension uses Armadillo's C++ headers, including template implementations.
The verified Windows build uses Armadillo 15.4.2 from the existing shared
RcppArmadillo installation. No Rcpp adapters or R library are linked.

The supplied Armadillo header notices state:

Copyright 2008-2016 Conrad Sanderson (<https://conradsanderson.id.au>)
and National ICT Australia (NICTA).

Armadillo is licensed under Apache License 2.0. The full text is in
`LICENSES/Apache-2.0.txt`. The RcppArmadillo `COPYRIGHTS` file separately identifies
its aggregation, integration, and packaging under GPL (>= 2); this extension
uses only the standalone Armadillo headers.

## pybind11

The extension includes pybind11 3.1.0 headers in the verified Windows build.
Copyright (c) 2016 Wenzel Jakob, all rights reserved. pybind11 is licensed under
BSD-3-Clause; the full supplied notice is in
`LICENSES/pybind11-BSD-3-Clause.txt`.

## Windows numerical and compiler libraries

The verified Windows extension links the existing Rtools45 static BLAS, LAPACK,
GNU C++, GNU Fortran, OpenMP, GCC support, quadmath, and MinGW runtime libraries.
It imports only the Python and Windows system runtime DLLs. Other platforms use
the system libraries chosen at build time.

BLAS and LAPACK use the modified BSD license. The upstream notice supplied with
the shared R distribution, including University of Tennessee, University of
California Berkeley, and University of Colorado Denver copyrights, is preserved
in `LICENSES/LAPACK-BSD.txt`.

GNU compiler runtime libraries are provided under GPL-3.0-or-later with GCC
Runtime Library Exception 3.1. The exception text is in
`LICENSES/GCC-exception-3.1.txt`; the GNU C++ header notice, including its
Hewlett-Packard and Silicon Graphics notices, is in
`LICENSES/libstdcxx-headers.txt`.

The installed quadmath header specifies GNU Library General Public License
version 2 or later, copyright 2010-2019 Free Software Foundation, Inc., written
by Francois-Xavier Coudert. Its notice and license are in
`LICENSES/libquadmath-notice.txt` and `LICENSES/LGPL-2.0.txt`.

The MinGW runtime uses Zope Public License 2.1 except for prominently marked
public domain, BSD or LGPL parts. The supplied notice is in
`LICENSES/mingw-w64.txt`. Windows pthread support includes the mingw-w64 and
Lockless Inc. notices preserved in `LICENSES/winpthreads.txt`.

## Runtime Python dependencies

NumPy and SciPy are installed separately and retain their own licenses. They
are dependencies rather than copied Python packages in this distribution.
The license texts above were copied from existing shared installations; no
private dependency or tool installation is included.
