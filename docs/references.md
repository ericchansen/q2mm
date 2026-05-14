# References

If you use Q2MM in your research, please cite the relevant publications below. Citing these works helps support continued development and gives proper credit to the contributors.

---

## Core method

1. **Norrby, P.-O.** Selectivity in Asymmetric Synthesis from QM-Guided Molecular Mechanics. *J. Mol. Struct. (THEOCHEM)* **2000**, *506*, 9–16. [DOI: 10.1016/S0166-1280(00)00398-5](https://doi.org/10.1016/S0166-1280(00)00398-5)

    Introduces the foundational Q2MM approach — using quantum mechanical reference data to parameterize molecular mechanics force fields for predicting stereoselectivity in asymmetric catalysis.

2. **Norrby, P.-O.** Deriving Force Field Parameters for Coordination Complexes. *Coord. Chem. Rev.* **2001**, *212*, 79–109. [DOI: 10.1016/S0010-8545(00)00296-4](https://doi.org/10.1016/S0010-8545(00)00296-4)

    Methodology for deriving MM force field parameters from QM data, covering coordination complexes and the gradient-based optimization approach used in Q2MM.

3. **O. Nilsson Lill, S.; Forbes, A.; Donoghue, P.; Verdolino, V.; Wiest, O.; Rydberg, P.; Norrby, P.-O.** Application of Q2MM to Stereoselective Reactions. *Curr. Org. Chem.* **2010**, *14*, 1629–1645. [DOI: 10.2174/138527210793563224](https://doi.org/10.2174/138527210793563224)

    Review of the Q2MM workflow — objective function construction, reference data types, and optimization strategies — with applications to stereoselective reactions.

4. **Limé, E.; Norrby, P.-O.** Improving the Q2MM Method for Transition State Force Field Modeling. *J. Comput. Chem.* **2015**, *36*, 244–250. [DOI: 10.1002/jcc.23797](https://doi.org/10.1002/jcc.23797)

    Introduces Hessian eigenvalue handling methods (C, D, and E) for transition states. q2mm implements curvature inversion (Method C).

5. **Hansen, E.; Rosales, A. R.; Tutkowski, B.; Norrby, P.-O.; Wiest, O.** Prediction of Stereochemistry using Q2MM. *Acc. Chem. Res.* **2016**, *49*, 996–1005. [DOI: 10.1021/acs.accounts.6b00037](https://doi.org/10.1021/acs.accounts.6b00037)

    Comprehensive review of Q2MM methodology, covering the theoretical framework, parameter optimization workflow, and successful predictions of stereochemical outcomes in catalytic reactions.

6. **Rosales, A. R.; Quinn, T. R.; Wahlers, J.; Tomberg, A.; Zhang, X.; Helquist, P.; Wiest, O.; Norrby, P.-O.** Application of Q2MM to Predictions in Stereoselective Synthesis. *Chem. Commun.* **2018**, *54*, 8294–8311. [DOI: 10.1039/C8CC03695K](https://doi.org/10.1039/C8CC03695K)

    Demonstrates application of Q2MM to predict stereoselectivity across diverse reaction types, validating the method's generality and predictive power.

---

## Seminario method

7. **Seminario, J. M.** Calculation of Intramolecular Force Fields from Second-Derivative Tensors. *Int. J. Quantum Chem.* **1996**, *60*, 1271–1277. [DOI: 10.1002/(SICI)1097-461X(1996)60:7<1271::AID-QUA8>3.0.CO;2-W](https://doi.org/10.1002/(SICI)1097-461X(1996)60:7%3C1271::AID-QUA8%3E3.0.CO;2-W)

    The foundational method for extracting bond and angle force constants from a QM Hessian matrix. Projects Cartesian second-derivative sub-blocks onto internal coordinates via eigendecomposition, producing initial force constant estimates without any MM calculations. Q2MM uses this as Stage 1 of the pipeline (see [`q2mm.models.seminario`](reference/q2mm/models/seminario.md)).

---

## QFUERZA

8. **Farrugia, M.; Helquist, P.; Norrby, P.-O.; Wiest, O.** Rapid FF Generation via Hessian-Informed Initial Parameters and Automated Refinement. *J. Chem. Theory Comput.* **2025**, *22*, 469–476. [DOI: 10.1021/acs.jctc.5c01751](https://doi.org/10.1021/acs.jctc.5c01751)

    Builds on the Seminario/FUERZA projection method by substituting known-problematic force constants — particularly hydrogen angle bends, which Seminario overestimates by ~2× — with empirical defaults (0.5 mdyn·Å/rad²). QFUERZA is the default strategy for `estimate_force_constants()`. Starting from QFUERZA parameters leads to faster optimizer convergence and fewer local-minimum traps. Tested on cis-platinum (ground state) and Rh-enamide (transition state). See [theory](how-it-works/theory.md#stage-1-qfuerza-estimation).

---

## MM3 force field

9. **Allinger, N. L.; Yuh, Y. H.; Lii, J. H.** Molecular Mechanics. The MM3 Force Field for Hydrocarbons. 1. *J. Am. Chem. Soc.* **1989**, *111*, 8551–8566. [DOI: 10.1021/ja00205a001](https://doi.org/10.1021/ja00205a001)

    Foundational MM3 force field paper. Q2MM uses the MM3 functional forms (cubic bond stretch, sextic angle bend) defined here; see `q2mm.models.constants`.

---

## Applications

10. **Donoghue, P. J.; Helquist, P.; Norrby, P.-O.; Wiest, O.** Development of a Q2MM Force Field for the Asymmetric Rhodium Catalyzed Hydrogenation of Enamides. *J. Chem. Theory Comput.* **2008**, *4*, 1313–1323. [DOI: 10.1021/ct800132a](https://doi.org/10.1021/ct800132a)

    The original Rh-enamide Q2MM force field — the primary benchmark system used throughout this codebase.

11. **Hansen, E.; Limé, E.; Norrby, P.-O.; Wiest, O.** Anomeric Effects in Sulfamides. *J. Phys. Chem. A* **2016**, *120*, 3677–3682. [DOI: 10.1021/acs.jpca.6b02757](https://doi.org/10.1021/acs.jpca.6b02757)

    Ground-state force field for sulfamides, using Q2MM to implicitly parameterize coupled anomeric effects via torsional energy scans. Demonstrates Q2MM beyond transition-state applications.

12. **Rosales, A. R.; Wahlers, J.; Limé, E.; Meadows, R. E.; Leslie, K. W.; Savin, R.; Bell, F.; Hansen, E.; Helquist, P.; Munday, R. H.; Wiest, O.; Norrby, P.-O.** Rapid Virtual Screening of Enantioselective Catalysts using CatVS. *Nat. Catal.* **2019**, *2*, 41–45. [DOI: 10.1038/s41929-018-0193-3](https://doi.org/10.1038/s41929-018-0193-3)

    Introduces CatVS, a virtual screening platform that uses Q2MM-derived transition state force fields to rapidly evaluate catalyst libraries for enantioselectivity.

13. **Burai Patrascu, M.; Pottel, J.; Pinus, S.; Bezanson, M.; Norrby, P.-O.; Moitessier, N.** From Desktop to Benchtop with Automated Computational Workflows for Computer-Aided Design in Asymmetric Catalysis. *Nat. Catal.* **2020**, *3*, 574–584. [DOI: 10.1038/s41929-020-0468-3](https://doi.org/10.1038/s41929-020-0468-3)

    Integrates Q2MM with machine learning to predict enantioselectivity for a broader range of reactions and catalyst types.

14. **Rosales, A. R.; Ross, S. P.; Helquist, P.; Norrby, P.-O.; Sigman, M. S.; Wiest, O.** Transition State Force Field for the Asymmetric Redox-Relay Heck Reaction. *J. Am. Chem. Soc.* **2020**, *142*, 9700–9707. [DOI: 10.1021/jacs.0c01979](https://doi.org/10.1021/jacs.0c01979)

    Develops a Q2MM transition state force field for the asymmetric redox-relay Heck reaction, accurately predicting both enantio- and site-selectivity.

15. **Wahlers, J.; Maloney, M.; Salahi, F.; Rosales, A. R.; Helquist, P.; Norrby, P.-O.; Wiest, O.** Stereoselectivity Predictions for the Pd-Catalyzed 1,4-Conjugate Addition. *J. Org. Chem.* **2021**, *86*, 5660–5667. [DOI: 10.1021/acs.joc.1c00136](https://doi.org/10.1021/acs.joc.1c00136)

    Applies Q2MM to predict stereoselectivity in palladium-catalyzed conjugate additions, demonstrating transferability to new reaction classes.

16. **Wahlers, J.; Margalef, J.; Hansen, E.; Bayesteh, A.; Helquist, P.; Diéguez, M.; Pàmies, O.; Wiest, O.; Norrby, P.-O.** Proofreading Experimentally Assigned Stereochemistry through Q2MM Predictions. *Nat. Commun.* **2021**, *12*, 6508. [DOI: 10.1038/s41467-021-27065-2](https://doi.org/10.1038/s41467-021-27065-2)

    Uses Q2MM predictions to identify and correct experimentally misassigned stereochemistry, demonstrating the method's value as a validation tool.

17. **Quinn, T. R.; Patel, H. N.; Koh, K. H.; Haines, B. E.; Norrby, P.-O.; Helquist, P.; Wiest, O.** Automated Fitting of Transition State Force Fields for Biomolecular Simulations. *PLOS ONE* **2022**, *17*, e0264960. [DOI: 10.1371/journal.pone.0264960](https://doi.org/10.1371/journal.pone.0264960)

    Extends Q2MM automation for biomolecular systems, enabling transition state force field fitting for enzymatic reactions.

18. **Wahlers, J.; Rosales, A. R.; Berkel, N.; Forbes, A.; Helquist, P.; Norrby, P.-O.; Wiest, O.** A Quantum-Guided Molecular Mechanics Force Field for the Ferrocene Scaffold. *J. Org. Chem.* **2022**, *87*, 12334–12341. [DOI: 10.1021/acs.joc.2c01553](https://doi.org/10.1021/acs.joc.2c01553)

    Develops specialized MM3* force field parameters for ferrocene-based ligands used in asymmetric catalysis.

19. **Maloney, M. P.; Stenfors, B. A.; Helquist, P.; Norrby, P.-O.; Wiest, O.** Interplay of Computation and Experiment in Enantioselective Catalysis. *ACS Catal.* **2023**, *13*, 14285–14299. [DOI: 10.1021/acscatal.3c03706](https://doi.org/10.1021/acscatal.3c03706)

    Reviews the synergy between computational (Q2MM) and experimental approaches in developing enantioselective catalysts.
