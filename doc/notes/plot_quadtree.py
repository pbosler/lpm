#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import argparse

def parser_setup():
  parser = argparse.ArgumentParser(description="setup quadtree plot options")
  parser.add_argument('filename', help="filename containing numpy data")
  return parser

if __name__ == '__main__':
  parser = parser_setup()
  args = parser.parse_args()
  if args.filename[-3:] != ".py":
    raise ValueError("expected a .py file")
  print("loading data from {}".format(args.filename))

  tcdata = __import__(args.filename[:-3])
  nverts = np.shape(tcdata.box_vert_crds)[0]
  nnodes = len(tcdata.node_keys)
  print(" ... {} vertices found.".format(nverts))

  fig, ax = plt.subplots()
  ax.plot(tcdata.box_vert_crds[:,0], tcdata.box_vert_crds[:,1], 'rs')
  boxes = []
  for i in range(nnodes):
    xy = tcdata.box_vert_crds[4*i,:]
    dx = tcdata.box_vert_crds[4*i+2,0] - tcdata.box_vert_crds[4*i,0]
    dy = tcdata.box_vert_crds[4*i+1,1] - tcdata.box_vert_crds[4*i,1]
    box = patches.Rectangle(xy, dx, dy, ec='r', fc=None)
    boxes.append(box)
  node_boxes = PatchCollection(boxes)
  node_boxes.set_array(tcdata.node_idxn)
  p = ax.add_collection(node_boxes)
  ax.plot(tcdata.points[:,0], tcdata.points[:,1], 'b.')
  fig.colorbar(p)
  plt.show()


