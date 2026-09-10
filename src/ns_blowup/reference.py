"""Independent CPU multiprecision reference for the implemented components.

Uses mpmath, not JAX, NumPy, or production quadrature/coordinate routines.
Values are mpmath numbers in a private context; decimal strings avoid rounding
inputs to binary64. Precision comparisons are empirical, not interval bounds.
Source: pinned paper (4.1) and Lemma A.6; see docs/multiprecision.md.
"""
import operator


class ComponentReference:
    """Reference on tau>0, radius>0, 0<h<.01, nu>0; no smooth axis model.

    Install with ``pip install 'ns-blowup[reference]'``. Each instance owns its
    precision context; constructing one does not change global mpmath precision.
    """

    def __init__(self, *, dps=50, h='0.005', nu='1', c='1'):
        try:
            from mpmath import mp
        except ImportError as exc:
            raise ImportError("Install the reference extra: pip install 'ns-blowup[reference]'") from exc
        self.dps = operator.index(dps)
        if self.dps < 15:
            raise ValueError('dps must be at least 15 decimal digits')
        self.mp = mp.clone()
        self.mp.dps = self.dps
        self.h = self._number(h, 'h')
        self.nu = self._positive(nu, 'nu')
        self.c = self._positive(c, 'c')
        if not 0 < self.h < self.mp.mpf('.01'):
            raise ValueError('h must lie strictly between 0 and .01')

    def _number(self, value, name):
        value = self.mp.mpf(value)
        if not self.mp.isfinite(value):
            raise ValueError(f'{name} must be finite')
        return value

    def _positive(self, value, name):
        value = self._number(value, name)
        if value <= 0:
            raise ValueError(f'{name} must be positive')
        return value

    def coordinates(self, radius, z, tau):
        """Return q, eta, X using bracketed bisection, independently of JAX."""
        m = self.mp
        radius = self._number(radius, 'radius')
        if radius < 0:
            raise ValueError('radius must be nonnegative')
        z, tau = self._number(z, 'z'), self._positive(tau, 'tau')
        D = m.mpf('.5') - self.h
        if not z:
            q = tau
        else:
            # Above this lower endpoint q-z²q^(2h) is strictly increasing.
            lo = max(tau, abs(z)**(1 / D))
            hi = 2 * lo
            while hi - z*z*hi**(2*self.h) < tau:
                hi *= 2
            for _ in range(m.prec + 8):
                mid = (lo + hi) / 2
                if mid - z*z*mid**(2*self.h) < tau:
                    lo = mid
                else:
                    hi = mid
            q = (lo + hi) / 2
        return {'q': q, 'eta': z / q**D, 'X': radius*radius / (2*q)}

    def heat_factor(self, Z, *, method='hyperu'):
        """H(Z) via Tricomi U or independent adaptive improper quadrature."""
        m = self.mp
        Z = self._number(Z, 'Z')
        if Z < 0:
            raise ValueError('Z must be nonnegative')
        if method not in ('hyperu', 'quadrature'):
            raise ValueError("method must be 'hyperu' or 'quadrature'")
        if not Z:
            return m.mpf(1)
        if method == 'hyperu':
            return Z**(-1-self.h) * m.hyperu(1+self.h, 2, 1/Z)
        return m.quad(lambda v: m.exp(-v)*v**self.h*(1+Z*v)**(-self.h),
                      [0, 1, m.inf]) / m.gamma(1+self.h)

    def swirl(self, radius, tau):
        """Azimuthal velocity K_nu; excludes the singular axis."""
        m = self.mp
        radius, tau = self._positive(radius, 'radius'), self._positive(tau, 'tau')
        s = radius*radius / (2*self.nu)
        return m.sqrt(self.nu)*self.c*s**(-m.mpf('.5')-self.h)*self.heat_factor(2*tau/s)

    def velocity(self, xyz, tau):
        """Cartesian exterior velocity at one 3D point."""
        if len(xyz) != 3:
            raise ValueError('xyz must have three coordinates')
        x, y, _ = (self._number(value, 'coordinate') for value in xyz)
        radius = self.mp.sqrt(x*x+y*y)
        k = self.swirl(radius, tau)
        return (-k*y/radius, k*x/radius, self.mp.mpf(0))

    def pressure(self, radius, tau):
        """Pressure normalized to zero at infinity, p=-integral_r^inf K²/s ds."""
        radius, tau = self._positive(radius, 'radius'), self._positive(tau, 'tau')
        # s=r/v maps infinity to an integrable endpoint on a finite interval.
        return -self.mp.quad(lambda v: self.swirl(radius/v, tau)**2/v if v else 0, [0, 1])

    def heat_residual(self, radius, tau):
        """K_t - nu*(K_rr + K_r/r - K/r²), with t=T-tau."""
        radius, tau = self._positive(radius, 'radius'), self._positive(tau, 'tau')
        dr = self.mp.diff(lambda r: self.swirl(r, tau), radius)
        drr = self.mp.diff(lambda r: self.swirl(r, tau), radius, 2)
        dt = -self.mp.diff(lambda q: self.swirl(radius, q), tau)
        return dt - self.nu*(drr + dr/radius - self.swirl(radius, tau)/radius**2)
