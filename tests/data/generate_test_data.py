import argparse
import numpy as np
import netCDF4 as nc
import time

def lat_lon_pts(nlon, verbose=False):
  nlat = nlon//2 + 1
  dlambda = 2*np.pi/nlon
  if verbose:
    print("generating (nlat, nlon) = ({}, {}) uniform longitude-latitude points".format(nlon, nlat))
  lon_vals = np.array([j*dlambda for j in range(nlon)])
  lat_vals = np.array([-0.5*np.pi + i*dlambda for i in range(nlat)])
  return np.squeeze(np.resize(lon_vals, (1,nlat*nlon))), np.squeeze(np.resize(lat_vals, (1,nlat*nlon)))

def get_parser():
  parser = argparse.ArgumentParser(
    prog="generate_test_data",
    description="generates data files for LPM software testing",
    epilog="running this program will overwrite any existing files"
  )
  parser.add_argument("-n", "--npoints", type=int, default=8,
    help="number of points for uniform grid data")
  parser.add_argument("-v", "--verbose", action="store_true",
    help="write extra info to console while running")
  parser.add_argument("-l", "--lat-lon-file", action="store_true",
    help="write lat-lon data file")
  parser.add_argument("-x", "--xy-file", action="store_true",
    help="write xy data files")
  return parser

def xy_pts(n, packed=True, verbose=False):
  x = np.linspace(-1,1, n);
  y = np.linspace(-1,1, n);
  if verbose:
    print("generating (nx, ny) = ({}, {}) uniform xy points".format(n, n))
  if packed:
    xx, yy = np.meshgrid(x,y)
    result = np.vstack([xx.ravel(), yy.ravel()])
  else:
    result = (np.squeeze(np.resize(x,(1,n*n))), np.squeeze(np.resize(y,(1,n*n))))
  return result

def xyz_sphere_pts(n, packed=False, verbose=False):
  lons, lats = lat_lon_pts(n)
  x = np.cos(lons)*np.cos(lats)
  y = np.sin(lons)*np.cos(lats)
  z = np.sin(lats)
  result = (x,y,z)
  return result

if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()

  nlon = args.npoints

  lons, lats = lat_lon_pts(nlon, args.verbose)
  ll_fname = "lat_lon_data.nc"
  if args.verbose:
    print("uniform lon-lat data will be saved to file: {}".format(ll_fname))
  ncll = nc.Dataset(ll_fname, "w")
  ncll.description = "LPM unit test lon/lat data."
  ncll.history = "created " + time.ctime(time.time())
  ncll.createDimension("nnodes", len(lons))
  ncll.createDimension("time", None)
  nclons = ncll.createVariable("lon", "f8", ("nnodes",))
  nclats = ncll.createVariable("lat", "f8", ("nnodes",))
  ncones = ncll.createVariable("ones", "f8", ("nnodes",))
  nclons.units = "radians east"
  nclats.units = "radians north"
  ncones.units = "null"
  nclons[:] = lons
  nclats[:] = lats
  ncones[:] = np.ones(len(lons), dtype=np.float64)

  ncll.close()

  xy_packed = xy_pts(args.npoints, packed=True, verbose=args.verbose)
  xyp_fname = "xy_packed.nc"
  if args.verbose:
    print("packed xy data will be saved to file: {}".format(xyp_fname))
  ncxyp = nc.Dataset(xyp_fname, "w")
  ncxyp.description = "LPM unit test packed xy data"
  ncxyp.history = "created " + time.ctime(time.time())
  ncxyp.createDimension("nnodes", args.npoints*args.npoints)
  ncxyp.createDimension("two", 2)
  ncxyp_xy = ncxyp.createVariable("xy", "f8", ("two","nnodes",))
  ncxyp_xy[:,:] = xy_packed
  ncxyp.close()

  xyu_fname = "xy_unpacked.nc"
  if args.verbose:
    print("unpacked xy data will be saved to file: {}".format(xyu_fname))
  xy_unpacked = xy_pts(args.npoints, packed=False, verbose=args.verbose)
  ncxyu = nc.Dataset(xyu_fname, "w")
  ncxyu.description = "LPM unit test unpacked xy data"
  ncxyu.history = "created " + time.ctime(time.time())
  ncxyu.createDimension("n_nodes", args.npoints*args.npoints)
  upx = ncxyu.createVariable("x", "f8", ("n_nodes"))
  upx[:] = xy_unpacked[0]
  upy = ncxyu.createVariable("y", "f8", ("n_nodes"))
  upy[:] = xy_unpacked[1]
  ncxyu.close()


