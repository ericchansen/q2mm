import copy
import logging
import logging.config
import os
import unittest
import numpy as np
from q2mm import linear_algebra

from q2mm.schrod_indep_filetypes import MM3, MacroModelLog, GaussLog, Mol2, mass_weight_hessian
from q2mm.seminario import seminario
from q2mm import constants as co
from q2mm import utilities

logger = logging.getLogger(__name__)





struct_path = "examples/rh-enamide/rh_enamide_training_set/rh_enamide_training_set.mol2"
qm_log_path = "examples/rh-enamide/rh_enamide_training_set/jaguar_spe_freq_in_out/*.01.in"
seminario_fld_path = "examples/rh-enamide/rh_hyd_enamide_seminario_q.fld"
mm_log_path = "examples/rh-enamide/mm3_results/q_seminario_fld/*.log"
ethane_og_fld_path = "test/ethane/start.fld"
mol2_path = "examples/rh-enamide/rh_enamide_training_set/rh_enamide_training_set.mol2"
def run_seminario(self):

    ethane_ff = seminario(self.ethane_og_fld, self.structs, self.hessians, zero_out=True, hessian_units=co.KJMOLA)
    ethane_ff.export_ff(self.seminario_fld_path, ethane_ff.params)
def read_in_files(self):
    self.ethane_og_fld = MM3(self.ethane_og_fld_path)
    self.mol2s = [Mol2(self.mol2_path)]
    self.structs = [mol2.structures[0] for mol2 in self.mol2s]
    self.qm_log = GaussLog(self.qm_log_path)
    self.qm_structs = self.qm_log.structures
    self.hessians = [struct.hess for struct in self.qm_structs]
def read_out_files(self):
    self.qm_log = GaussLog(self.qm_log_path, au_hessian=False)
    self.qm_structs = self.qm_log.structures
    self.mm_log = MacroModelLog(self.mm_log_path)
def test_seminario_ethane_hessian(self):
    self.read_in_files()
    self.read_out_files()
    qm_hessian = self.qm_structs[0].hess # KJMOLA
    mass_weight_hessian(qm_hessian, self.structs[0].atoms)
    mm_hessian = self.mm_log.hessian
    mm_hessian = mm_hessian #* co.HARTREE_TO_KJMOL / co.HARTREE_TO_KCALMOL
    diff = mm_hessian - qm_hessian
    proportion_error = diff / qm_hessian
    percent_error_signed = proportion_error * 100
    percent_error = np.abs(percent_error_signed)
    max = np.max(percent_error)
    min = np.min(percent_error)
    avg = np.average(percent_error)
    median = np.median(percent_error)
    stddev = np.std(percent_error)
    mm_eigvals, mm_eigvecs = np.linalg.eigh(self.mm_log.hessian)
    reformed_mm_hessian = linear_algebra.reform_hessian(mm_eigvals, self.qm_log.evecs)
    # reformed_diff = reformed_mm_hessian - qm_hessian
    # reformed_proportion_error = reformed_diff / qm_hessian
    # reformed_percent_error_signed = reformed_proportion_error * 100
    # reformed_percent_error = np.abs(reformed_percent_error_signed)
    # reformed_max = np.max(reformed_percent_error)
    # reformed_min = np.min(reformed_percent_error)
    # reformed_avg = np.average(reformed_percent_error)
    # reformed_median = np.median(reformed_percent_error)
    # reformed_stddev = np.std(reformed_percent_error)
    mm_eigenmatrix = np.dot(np.dot(self.qm_log.evecs, mm_hessian), self.qm_log.evecs.T)
    qm_evals = self.qm_log.evals * co.HESSIAN_CONVERSION
    qm_eigenmatrix = np.diag(qm_evals)
    mm_eig_diag = np.diag(mm_eigenmatrix)
    mm_eig_tril = np.tril(mm_eigenmatrix)
    qm_eig_tril = np.tril(qm_eigenmatrix)
    np.savetxt("test/ethane/qm_eig_tril.csv", qm_eig_tril, delimiter=',')
    np.savetxt("test/ethane/mm_eig_tril.csv", mm_eig_tril, delimiter=',')
