import numpy as np
def diffusion(w, h, dx, dy, D, Tcool, Thot, r, cx, cy, nsteps):
    nx, ny = int(w/dx), int(h/dy)

    dx2, dy2 = dx*dx, dy*dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

    u0 = Tcool * np.ones((nx, ny))
    u = u0.copy()

    r2 = r**2
    for i in range(nx):
        for j in range(ny):
            p2 = (i*dx-cx)**2 + (j*dy-cy)**2
            if p2 < r2:
                u0[i,j] = Thot

    def do_timestep(u0, u):
        # Propagate with forward-difference in time, central-difference in space
        u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
              (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
              + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )

        u0 = u.copy()
        return u0, u

    u_f = []
    for m in range(nsteps):
        u0, u = do_timestep(u0, u)
        u_f.append(u0)
    return u_f