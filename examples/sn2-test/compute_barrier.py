"""Compute F- energy and barrier height for literature comparison."""
import psi4
psi4.set_memory('1 GB')
psi4.core.set_output_file('NUL', False)

f_minus = psi4.geometry("""
    -1 1
    F 0.0 0.0 0.0
""")
e = psi4.energy('b3lyp/6-31G*', molecule=f_minus)
print(f'F- energy at B3LYP/6-31G*: {e:.12f} Ha')

ts_e = -239.522474335547
ch3f_e = -139.733945284233
barrier_ha = ts_e - (ch3f_e + e)
barrier_kcal = barrier_ha * 627.509
print(f'CH3F energy: {ch3f_e:.12f} Ha')
print(f'TS energy: {ts_e:.12f} Ha')
print(f'Barrier: {barrier_kcal:.2f} kcal/mol')
