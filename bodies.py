import numpy as np
from constants import *


class System:
    """
    Object containing star and planets.
    """
    def __init__(self, star, planets):
        """
        Parameters
        ----------
        star: Star
        planets: Planet or list of Planets
        """
        self.star = star

        # Force list, in case fed a single planet
        if type(planets) is Planet:
            planets = [planets]

        periods = [pl.period for pl in planets]
        planets_sorted = [pl for _, pl in sorted(zip(periods, planets))]
        self.planets = planets_sorted  # self.planets are sorted in order of increasing period

        star.set_system(self)
        for rank, pl in enumerate(planets):
            rank += 1  # rank should start at 1, not 0.
            pl.fill_in_info(self, rank)


class Star:
    """
    Object containing star parameters.
    """
    def __init__(self, mass=1.0, v0=0., t0=0., name='', ra=None, dec=None):
        """
        Parameters
        ----------
        mass: float
            Solar masses
        v0: float
            Offset velocity of star (m/s)
        t0: float
            JD of start of observations
        name: str
            Name of star
        ra: None or [TBD]
            Right ascension
        dec: None or [TBD]
            Declination
        """
        self.mass = mass
        self.name = name
        self.ra = ra
        self.dec = dec
        self.system = None
        self.v0 = v0
        self.t0 = t0

    def set_system(self, system):
        """Attach System to this Star"""
        self.system = system


class Planet:
    """
    Object containing planet parameters.
    """
    def __init__(self, period, ecc=0., incl=90., w=None, tp=None,
                 mass=None, K=None, sma=None, name='', is_real=True, random=False):
        """
        Parameters
        ----------
        period: float
            days
        ecc: float
        incl: float
            degrees, with 90 degrees being transiting
        w: float
            radians
        tp: float
            JD of periastron passage
        mass: float or None
            M_Earth
        K: float or None
            Doppler semi-amplitude (m/s)
        sma: float or None
            Semi-major axis (AU)
        name: str
            If blank, is set to star's name
        is_real: bool
            is planet is "real", or is it a fitted planet candidate
        random: bool
            If set, then unspecified orbital parameters (w and tp) are set randomly.
        """
        self.period = period
        self.ecc = ecc
        self.incl = incl
        self.w = w
        self.tp = tp
        self.mass = mass
        self.K = K
        self.sma = sma
        self.name = name
        self.is_real = is_real
        self.random = random
        self.rank = None

        self.sini = np.sin(incl * np.pi / 180.)

        # Initialize attributes that are set later
        self.system = None
        self.mass_star = None

    def fill_in_info(self, system, rank):
        """Called once star and system have been attached."""

        self.system = system

        if self.w is None:
            if self.random:
                self.w = 2 * np.pi * np.random.random()
            else:
                self.w = 0.

        if self.tp is None:
            self.tp = system.star.t0 + (self.period * np.random.random())

        if self.mass is None:
            # Determine mass in mEarth
            assert type(self.K) is float or type(self.K) is int or type(self.K) is np.float64, "Planet semi-amplitude, K, was not set."
            self.mass = (2. * np.pi * G / (self.period * day)) ** (-1./3) * (self.K * np.sqrt(1 - self.ecc ** 2.)) * (
                            1./(mEarth * self.sini)) * ((self.system.star.mass * mSun) ** (2./3))

        if self.K is None:
            # Determine semi-amplitude in m/s
            assert type(self.mass) is float or type(self.mass) is int or type(self.mass) is np.float64, "Planet mass was not set."
            self.K = (2. * np.pi * G / (self.period * day)) ** (1./3) * (self.mass * mEarth * self.sini) * (
                        (self.system.star.mass * mSun) ** (-2./3)) * (1./np.sqrt(1 - self.ecc ** 2.))

        if self.sma is None:
            # Determine semi-major axis in AU
            total_mass = (self.mass * mEarth) + (self.system.star.mass * mSun)
            self.sma = (1 / au) * (((self.period * year) ** 2 * G * total_mass) / (4. * np.pi * np.pi)) ** (1./3)

        self.rank = rank

        if self.name is '':
            star_name = self.system.star.name
            if star_name is '':
                self.name = 'b'
            else:
                self.name = star_name + ' b'

        # self.params = [self.period, self.ecc, self.tp, self.w, self.K, self.mass]
        # self.params_print = [self.period, self.ecc, self.tp - self.t0, self.w * 180. / pi, self.K, self.mass]
