# Growi Oracle Values

This file records the external Growi 1.0.3 checks used as ground-truth
regression data.  The Rust crate does not shell out to Growi in normal tests;
the values below are hardcoded in `src/growi_oracle.rs`.

## Isolated Build

The package was downloaded from:

- `https://agag-gathmann.math.rptu.de/de/growi.php`
- `https://agag-gathmann.math.rptu.de/pub/growi-1.0.3.tar.gz`

It was unpacked and built under `/tmp/growi-work`, with local dependencies
installed into `/tmp/growi-work/install`:

- GMP 6.3.0 from `https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz`
- GDBM 1.23 from `https://ftp.gnu.org/gnu/gdbm/gdbm-1.23.tar.gz`
- M4 1.4.19 from `https://ftp.gnu.org/gnu/m4/m4-1.4.19.tar.gz`

Growi needed `-fpermissive` with the current system C++ compiler because the
old headers contain a friend declaration with a default argument.  The
installed wrapper at `/tmp/growi-work/install/bin/growi` was patched in `/tmp`
only to add this flag.

Runtime environment used for the checks:

```sh
PATH=/tmp/growi-work/install/bin:/usr/bin:/bin
LD_LIBRARY_PATH=/tmp/growi-work/install/lib
CPLUS_INCLUDE_PATH=/tmp/growi-work/install/include
LIBRARY_PATH=/tmp/growi-work/install/lib
```

The README smoke checks passed:

```text
growi lines in quintic threefold
2875

growi G=2,D=5 in P^4 thru H^3*psi^22
-41369/110075314176

growi elliptic cubics in P^2 thru H^2:9
1
```

## Recorded Projective-Space Values

```text
growi G=2,D=3 in P^2 thru psi^11
163/41472

growi G=2,D=3 in P^2 thru H*psi^10
-421/207360

growi G=2,D=3 in P^2 thru H^2*psi^9
11/17280

growi G=1,D=1 in P^1 thru H*psi^2
1/24

growi G=1,D=1 in P^1 thru psi^3
0

growi G=2,D=3 in P^1 thru H*psi^8
23/41472

growi G=2,D=3 in P^1 thru psi^9
-977/622080

growi elliptic cubics in P^2 thru H^2:9
1

growi G=2,D=5 in P^2 thru H^2:16
36855

growi G=3,D=5 in P^2 thru H^2:17
7915

growi G=2,D=3 in P^3 thru H^2:12
5930

growi G=2,D=5 in P^4 thru H^3*psi^22
-41369/110075314176
```

The two values that exposed the current bug are:

```text
q^3 [tau9(H^2)] = 11/17280
q^3 [tau10(H)] = -421/207360
```

Before the R-recursion fix, the local S/R graph path returned
`-878851/69120` and `1066619/207360` for those two invariants.

The bug was in the R-matrix diagonal recursion: off-diagonal entries were
computed from flatness, but diagonal entries were not integrated from the
diagonal flatness equation.  Odd diagonal entries were frozen at their
classical constants, and even diagonals were filled from unitarity.  The fixed
recursion solves every diagonal q-series from
`q d R_k / dq + V R_k = 0` on the diagonal, using the classical value only as
the q^0 integration constant.
