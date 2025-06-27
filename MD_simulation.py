import numpy as np
import time
import tkinter as tk
import csv
import os
import json
import shutil
#Important: all the units are reduced to 1 like energy scale epsilon, lenght scale sigma nad mass

def setlat(nx, ny, nz, vol, npart):

    # Generates a 3D face-centered cubic (FCC) crystal of nx*ny*nz unit cells, each containing 4 particles.

    a0 = (vol / (nx * ny * nz)) ** (1/3)  # unit cell size

    x0 = []
    y0 = []
    z0 = []

    i = 0
    xcm0 = 0.0
    ycm0 = 0.0
    zcm0 = 0.0

    for iz in range(1, 2 * nz + 1):
        for iy in range(1, 2 * ny + 1):
            for ix in range(1, 2 * nx + 1):
                if (ix + iy + iz) % 2 == 0:
                    i += 1
                    x = a0 * ix + 0.5 * a0 * ((iy + iz) % 2)
                    y = a0 * iy + 0.5 * a0 * ((ix + iz) % 2)
                    z = a0 * iz + 0.5 * a0 * ((ix + iy) % 2)

                    x0.append(x)
                    y0.append(y)
                    z0.append(z)

                    xcm0 += x
                    ycm0 += y
                    zcm0 += z

    # Convert to numpy arrays
    x0 = np.array(x0)
    y0 = np.array(y0)
    z0 = np.array(z0)

    # center of mass
    xcm0 /= i
    ycm0 /= i
    zcm0 /= i

    return x0, y0, z0, xcm0, ycm0, zcm0

def initv(temp, npart, D, dt, x0, y0, z0):
    # initial Gaussian velocities
    vx = np.random.normal(0.0, 1.0, npart)
    vy = np.random.normal(0.0, 1.0, npart)
    vz = np.random.normal(0.0, 1.0, npart)

    # remove center of mass velocity
    vx -= np.mean(vx)
    vy -= np.mean(vy)
    vz -= np.mean(vz)

    # recaling to get temp
    sumv2 = np.sum(vx**2 + vy**2 + vz**2)
    nf = D * (npart - 1) - 1
    fs = np.sqrt(temp / (sumv2 / nf))

    vx *= fs
    vy *= fs
    vz *= fs

    # estimate previous positions
    x_prev = x0 - vx * dt
    y_prev = y0 - vy * dt
    z_prev = z0 - vz * dt

    return vx, vy, vz, x_prev, y_prev, z_prev, nf

def LJ_shift(rc2):
    r2i = 1 / rc2
    r6i = r2i ** 3
    r12i = r6i ** 2
    return 4 * (r12i - r6i)

def LJ_potential_and_force(r2, rc2):
    shift = LJ_shift(rc2)
    epsilon = 1e-12
    r2i = 1 / (r2 + epsilon)
    r6i = r2i ** 3
    r12i = r6i ** 2
    U = 4 * (r12i - r6i) - shift
    f = 24 * r2i * (2 * r12i - r6i)
    return U, f

def WF_potential_and_force(r2, rc2):
    epsilon = 1e-12
    r2i = 1 / (r2 + epsilon)
    r2im1 = r2i - 1.0
    rc2r2im1 = rc2 * r2i - 1.0
    U = r2im1 * rc2r2im1**2
    f = 6.0 * r2i**2 * rc2r2im1 * (rc2r2im1 - 2)

    return U, f

# FandE made by me


# def FandE(rc, x0, y0, z0, box, potential_type="LJ"):
#     rc2 = rc ** 2
#     npart = len(x0)
#
#     fx = np.zeros(npart)
#     fy = np.zeros(npart)
#     fz = np.zeros(npart)
#
#     en = 0.0
#
#     for i in range(npart - 1):
#         for j in range(i + 1, npart):
#             dx = x0[i] - x0[j]
#             dy = y0[i] - y0[j]
#             dz = z0[i] - z0[j]
#
#             dx -= box * np.round(dx / box)
#             dy -= box * np.round(dy / box)
#             dz -= box * np.round(dz / box)
#
#             r2 = dx**2 + dy**2 + dz**2
#
#             if r2 < rc2:
#                 if potential_type == "LJ":
#                     U, f = LJ_potential_and_force(r2)
#                 elif potential_type == "WF":
#                     U, f = WF_potential_and_force(r2, rc2)
#
#                 fxij = f * dx
#                 fyij = f * dy
#                 fzij = f * dz
#
#                 # Newton s 3rd law
#                 fx[i] += fxij
#                 fy[i] += fyij
#                 fz[i] += fzij
#
#                 fx[j] -= fxij
#                 fy[j] -= fyij
#                 fz[j] -= fzij
#
#                 en += U
#
#     return fx, fy, fz, en
#



# vectorized by chatgpt

def FandE_vectorized(rc, x0, y0, z0, box, potential_type="LJ"):
    rc2 = rc ** 2
    positions = np.stack([x0, y0, z0], axis=1)  # shape (npart, 3)
    npart = positions.shape[0]

    # Pairwise displacement vectors
    rij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n, n, 3)

    # Minimum image convention
    rij -= box * np.round(rij / box)

    # Pairwise squared distances
    r2 = np.sum(rij ** 2, axis=-1)  # shape (n, n)

    # Upper triangle indices
    i_indices, j_indices = np.triu_indices(npart, k=1)

    # Apply cutoff mask
    r2_pairs = r2[i_indices, j_indices]
    epsilon = 1e-12
    mask = (r2_pairs < rc2) & (r2_pairs > epsilon)

    rij_pairs = rij[i_indices[mask], j_indices[mask]]  # shape (pairs, 3)
    r2_valid = r2_pairs[mask]

    # Select potential type
    if potential_type == "LJ":
        U, f_scalar = LJ_potential_and_force(r2_valid, rc2)
    elif potential_type == "WF":
        U, f_scalar = WF_potential_and_force(r2_valid, rc2)
    else:
        raise ValueError("Unknown potential type: use 'LJ' or 'WF'.")

    # Compute force vectors
    fij = f_scalar[:, np.newaxis] * rij_pairs  # shape (pairs, 3)

    # Initialize forces
    forces = np.zeros_like(positions)

    # Accumulate forces using Newton's third law
    np.add.at(forces, i_indices[mask], fij)
    np.subtract.at(forces, j_indices[mask], fij)

    # Total potential energy
    en = np.sum(U)

    # Return component-wise forces and energy
    fx, fy, fz = forces[:, 0], forces[:, 1], forces[:, 2]
    return fx, fy, fz, en

def IntegrateV(x0, y0, z0, x_prev, y_prev, z_prev, fx, fy, fz, dt):
    # Verlet position update
    x_next = 2 * x0 - x_prev + dt ** 2 * fx
    y_next = 2 * y0 - y_prev + dt ** 2 * fy
    z_next = 2 * z0 - z_prev + dt ** 2 * fz


    vxi = (x_next - x_prev) / (2 * dt)
    vyi = (y_next - y_prev) / (2 * dt)
    vzi = (z_next - z_prev) / (2 * dt)


    return x0, y0, z0, x_next, y_next, z_next, vxi, vyi, vzi

def sample(vxi, vyi, vzi, nf, npart, en):
    sumv = 0
    sumv2 = 0

    sumv2 = np.sum(vxi**2 + vyi**2 + vzi**2)
    temp = sumv2 / nf

    etot = (en + 0.5 * sumv2)
    epart = (en + 0.5 * sumv2) / npart

    return temp, etot, epart


# """"
# program MD basic MD code
# [...]
# setlat function to initialize positions x
# initv(temp) function to initialize velocities vx
# t=0
# while (t < tmax) do main MD loop
# FandE function to compute forces and total energy
# Integrate-V function to integrate equations of motion
# t=t+delt update time
# sample function to sample averages
# enddo
# end program
# """""




import os
import csv
import time
import numpy as np

def MD_simulation(nx, ny, nz, density, temp, dt, tmax, rc, potential_type, sample_interval, total_save_interval):
    D = 3
    npart = 4 * nx * ny * nz
    vol = npart / density
    box = vol ** (1/3)

    os.makedirs("states", exist_ok=True)

    x0, y0, z0, xcm0, ycm0, zcm0  = setlat(nx, ny, nz, vol, npart)
    vx, vy, vz, x_prev, y_prev, z_prev, nf = initv(temp, npart, D, dt, x0, y0, z0)

    t = 0
    step = 1
    total_time = 0
    output_csv_path = "log.csv"

    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "t", "total_time", "temp", "etot", "epart"])

        while t <= tmax:
            start = time.time()
            fx, fy, fz, en = FandE_vectorized(rc, x0, y0, z0, box, potential_type)
            x_prev, y_prev, z_prev, x0, y0, z0, vxi, vyi, vzi = IntegrateV(x0, y0, z0, x_prev, y_prev, z_prev, fx, fy, fz, dt)

            if step % sample_interval == 0:
                temp, etot, epart = sample(vxi, vyi, vzi, nf, npart, en)

                end = time.time()
                elapsed_time = end - start
                total_time += elapsed_time

                progress = round(t / tmax * 100)
                print(f"Step {step}, time {t:.2f}/{tmax} ({progress}%)")

                writer.writerow([step, t, total_time, temp, etot, epart])
                csvfile.flush()
                os.fsync(csvfile.fileno())  # Force disk write

            if step % total_save_interval == 0:
                np.savez(f"states/state_step_{step}.npz",
                         x0=x0, y0=y0, z0=z0,
                         vx=vxi, vy=vyi, vz=vzi)

            step += 1
            t += dt

    print("Simulazione Conclusa")


with open("input_config.json", "r") as f:
    parameters = json.load(f)

# Assign each parameter to a variable
temp = parameters["temp"]
dt = parameters["dt"]
tmax = parameters["tmax"]
sample_interval = parameters["sample_interval"]
total_save_interval = parameters["total_save_interval"]
density = parameters["density"]
rc = parameters["rc"]
nx = parameters["nx"]
ny = parameters["ny"]
nz = parameters["nz"]
potential_type = parameters["potential_type"]



MD_simulation(nx, ny, nz, density, temp, dt, tmax, rc, potential_type, sample_interval, total_save_interval)



