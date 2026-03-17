# This file contains all of the most updated atom types needed to use any Q2MM
# FF.

# For descriptions of the columns, please see the atom type file atom.typ that
# is distributed with any Schrodinger release.

# Try to use these general types in new FFs if possible. Make note of the charge
# state. If that's not right, you will have to make a new atom type (see atom
# type 250 for an example).

#&Palladium (Heck, Palladium allyl)
201      46  106.42         Pd    20     1.70      T      F      !  2.2       T       !     !         !      !   +1  NCA       2   !
#&Ru (Heck)
202      44  101.07         Ru    20     1.70      T      F      !  2.2       T       !     !         !      !    0  NCA       4   !
#&Rh (Heck)
203      45  102.91         Rh    20     1.70      T      F      !  2.2       T       !     !         !      !    0  NCA       4   !
#&Ir (Heck)
204      77  192.22         Ir    20     1.70      T      F      !  2.2       T       !     !         !      !    0  NCA       4   !
#&Dummy atom (Ferrocene)
205      -2    0.0000       D1    10     1.00      F      F      !  0.0       T       !     !         !      !    0  NCA       1   !
#$Osmium (Sharpless AD)
206      76  190.23         Os    20     1.75      T      F      !  1.5       T       !     !         !      !    0  NCA       1   !

# Atom types below are specific for the Rh catalyzed hydrogenation of enamides.
# Technically, only the definition for Rh is required (needed the +1 charge).
# The other atom types are depreciated, but are included such that older
# versions of Donoghue's TSFF can still be used.

#&Rh catalyzed hydrogenation of enamides: Rh
250        !    102.90550    RH     4     1.75      T      F      !   1.50     T       !     !         !      !   +1  NCA       1   !
#&Rh catalyzed hydrogenation of enamides: H-moving hydrogen
251        1      1.00797    H7    21     1.20      F      F      !   2.00     F       !     !         !      !    0  NCA       2   !
#&Rh catalyzed hydrogenation of enamides: H-stationary hydrogen
252        1      1.00797    H6    21     1.20      F      F      !   2.00     F       !     !         !      !    0  NCA       1   !
#&Rh catalyzed hydrogenation of enamides: Phosphorus trans-O
253      15     30.97380    P2    15     1.80      T      F      !   2.10     F       !     !         !      !    0  TET       3   H1
#&Rh catalyzed hydrogenation of enamides: Phosphorus cis-O
254      15     30.97380    P1    15     1.80      T      F      !   2.10     F       !     !         !      !    0  TET       3   H1

# Atom types below are specific for ruthenium catalyzed ketone hydrogenation.
# Stereoselectivity in Asymmetric Catalysis: The Case of Ruthenium-Catalyzed
# Ketone Hydrogenation. Limé, E.; Lundholm, M.D.; Forbes, A.; Wiest, O.;
# Helquist, P.; Norrby, P.-O. J. Chem. Theory Comput., 2014, 10, 2427--2435.

# Currently, this Z0 is still the same as the unmodified Z0.

62        !    100.00000    Z0     4     1.75      T      F      !   1.50     T       !     !         !      !    0  NCA       1   !
