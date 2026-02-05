import resolvent4py as res4py
from petsc4py import PETSc
import numpy as np


path = "jacobian/"
freqs = np.load(path + "freqs.npy")
rows = np.load(path + "rows.npy")
cols = np.load(path + "cols.npy")
vals = [np.load(path + "vals_%02d.npy" % j) for j in range(len(freqs))]

for j in range(len(vals)):
    arrays = [rows, cols, vals[j]]
    fnames = ["rows_%02d.dat" % j, "cols_%02d.dat" % j, "vals_%02d.dat" % j]
    for i, array in enumerate(arrays):
        fname = path + fnames[i]
        vec = PETSc.Vec().createWithArray(
            array, len(array), None, PETSc.COMM_SELF
        )
        res4py.write_to_file(fname, vec)
        vec.destroy()

rows_id = np.load(path + "rows_id.npy")
cols_id = np.load(path + "cols_id.npy")
vals_id = np.load(path + "vals_id.npy")

arrays = [rows_id, cols_id, vals_id]
fnames = ["rows_id.dat", "cols_id.dat", "vals_id.dat"]
for i, array in enumerate(arrays):
    fname = path + fnames[i]
    vec = PETSc.Vec().createWithArray(array, len(array), None, PETSc.COMM_SELF)
    res4py.write_to_file(fname, vec)
    vec.destroy()
