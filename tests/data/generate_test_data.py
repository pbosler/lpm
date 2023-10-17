import argparse
import numpy as np
import netCDF4 as nc
import time

def lat_lon_pts(nlon, verbose=False):
  nlat = nlon//2 + 1
  dlambda = 2*np.pi/nlon
  if verbose:
    print("generating (nlat, nlon) = ({}, {}) uniform longitude-latitude points".format(nlon, nlat))
  lons = np.array([j*dlambda for j in range(nlon)])
  lats = np.array([-0.5*np.pi + i*dlambda for i in range(nlat)])
  return lons, lats

def get_parser():
  parser = argparse.ArgumentParser(
    prog="generate_test_data",
    description="generates data files for LPM software testing",
    epilog="running this program will overwrite any existing files"
  )
  parser.add_argument("-nl", "--nlon", type=int, default=10,
    help="number of longitude points for uniform lon-lat data")
  parser.add_argument("-v", "--verbose", action="store_true",
    help="write extra info to console while running")
  return parser


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()

  nlon = args.nlon

  lons, lats = lat_lon_pts(nlon, args.verbose)
  ll_fname = "lat_lon_data.nc"
  if args.verbose:
    print("uniform lon-lat data will be saved to file: {}".format(ll_fname))
  ncll = nc.Dataset(ll_fname, "w")
  ncll.description = "Unit test data."
  ncll.history = "created " + time.ctime(time.time())
  ncll.createDimension("lon", nlon)
  ncll.createDimension("lat", len(lats))
  ncll.createDimension("time", None)
  nclons = ncll.createVariable("lon", "f4", ("lon",))
  nclats = ncll.createVariable("lat", "f4", ("lat",))
  nclons.units = "radians east"
  nclats.units = "radians north"
  nclons[:] = lons
  nclats[:] = lats

  ncll.close()
