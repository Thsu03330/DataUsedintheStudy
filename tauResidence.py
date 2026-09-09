import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# ==============================================
# User input parameters
# ==============================================
topology_file = "system.data"                  
trajectory_file = "dump.lammpstrj" 
trajectory_format = "LAMMPSDUMP"               
dt = 0.1                                            # time in p
center_sel = "type X"                               # Define the central atom
ligand_sel = "type Y"                               # Define coordination atom
r_cutoff = 3.25                                     # Define radius in A
# Convergence parameters
convergence_tol = 0.02                         # Allowable tail relative deviation
tail_fraction = 0.3                            # Proportion of tail frames used for convergence check
# ==============================================
u = mda.Universe(topology_file, trajectory_file,
                 format=trajectory_format, dt=dt)
center_atoms = u.select_atoms(center_sel)
ligand_atoms = u.select_atoms(ligand_sel)

if len(center_atoms) == 0:
    raise ValueError("No center atoms selected, please check 'center_sel'.")
if len(ligand_atoms) == 0:
    raise ValueError("No ligand atoms selected, please check 'ligand_sel'.")

print(f"Number of center atoms: {len(center_atoms)}")
print(f"Number of ligand atoms: {len(ligand_atoms)}")

n_frames = len(u.trajectory)
n_ligands = len(ligand_atoms)

h_matrix = np.zeros((n_ligands, n_frames), dtype=np.int8)

for ts_idx, ts in enumerate(u.trajectory):
    dist = mda.lib.distances.distance_array(ligand_atoms.positions,
                                            center_atoms.positions,
                                            box=ts.dimensions)
    in_shell = np.any(dist <= r_cutoff, axis=1)
    h_matrix[:, ts_idx] = in_shell.astype(np.int8)

max_tau = n_frames - 1
C_t = np.zeros(max_tau)
for tau in range(max_tau):
    corr_sum = 0.0
    for h in h_matrix:
        h0 = h[:n_frames - tau]
        ht = h[tau:]
        corr_sum += np.sum(h0 * ht) / len(h0)
    C_t[tau] = corr_sum / n_ligands

p = h_matrix.mean()                     # <h_i>
C0 = C_t[0]
if C0 - p**2 == 0:
    raise ValueError("C(0) - <h>^2 is zero, cannot normalize; check your data.")
C_norm = (C_t - p**2) / (C0 - p**2)

tail_len = max(1, int(max_tau * tail_fraction))
C_tail_mean = C_t[-tail_len:].mean()
deviation = abs(C_tail_mean - p**2) / (C0 - p**2)
if deviation > convergence_tol:
    print(f"Warning: C_R(t) may not be converged. "
          f"Tail relative deviation = {deviation:.4f} > tolerance {convergence_tol}")
    print("       Simulation length may be insufficient; lifetime may be underestimated.")
else:
    print(f"C_R(t) is well converged (tail relative deviation = {deviation:.4f} <= {convergence_tol})")

time = np.arange(max_tau) * u.trajectory.dt

tau_residence = np.trapezoid(C_norm, time)
print(f"First shell residence time τ_Residence = {tau_residence:.3f} ps")

np.savetxt("Results_residence.dat", np.column_stack((time, C_norm)),
           header="Time(ps)  C_norm(t)  (de-centered and normalized)")
with open("Results_residence.dat", "a") as f:
    f.write(f"\n# Summary\n")
    f.write(f"# Center selection: {center_sel}\n")
    f.write(f"# Ligand selection: {ligand_sel}\n")
    f.write(f"# Shell radius r_cutoff: {r_cutoff} A\n")
    f.write(f"# Number of ligand atoms: {n_ligands}\n")
    f.write(f"# <h_i> = {p:.4f}\n")
    f.write(f"# C_R(0) = {C0:.4f}\n")
    f.write(f"# Residence time t_Residence (ps): {tau_residence:.3f}\n")

plt.figure(figsize=(6, 5))
plt.plot(time, C_norm, linewidth=2)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel("Time (ps)")
plt.ylabel("$C_{norm}(t)$")
plt.title("First coordination shell residence time (de-centered & normalized)")
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig("Residence_time.png", dpi=600)
plt.show()
