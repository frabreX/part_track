import numpy as np

""""
program MD basic MD code
[...]
setlat function to initialize positions x
initv(temp) function to initialize velocities vx
t=0
while (t < tmax) do main MD loop
FandE function to compute forces and total energy
Integrate-V function to integrate equations of motion
t=t+delt update time
sample function to sample averages
enddo
end program
"""""

global tmax, delt, temp
tmax = 10
delt = 10 ** (-12)
temp = 300



nx = 3
ny = 3
nz = 3

vol = 9




def setlat(nx, ny, nz, vol):

    # Generates a 3D face-centered cubic (FCC) crystal of nx*ny*nz unit cells, each containing 4 particles.

    npart = 4 * nx * ny * nz
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

# x0, y0, z0, xcm0, ycm0, zcm0 = setlat(nx, ny, nz, vol)
#
#
# print(x0.shape)


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
    x_prev = x - vx * dt
    y_prev = y - vy * dt
    z_prev = z - vz * dt

    return vx, vy, vz, x_prev, y_prev, z_prev



def LJ_potential():

def WF_potential():




def FandE(rc, x0, y0, z0, box):

    rc2 = rc ** 2

    fx = np.zeros_like(x0)
    fy = np.zeros_like(y0)
    fz = np.zeros_like(z0)

    en = 0

    #the idea behind: i could do a for loop to match all the pair of particles:
    #for 1 ≤ i ≤ npart-1 do
    # for i+1 ≤ j ≤ npart do loop over all pairs
    # but it will recquire a lot of computation and time as it is python
    #therefore i look for a numpy solution

    for i in range(0, len(x0)+1):

        temp = np.full_like(x0, i)
        xr = temp - x0
        x0 = np.delete(x0, 0)

        xr = xr - box * (xr / box)

        r2 = xt ** 2

        if r2 < rc2:
            LJ_potential(r2)
            # WF_potential(r2)


