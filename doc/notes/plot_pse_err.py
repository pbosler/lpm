import numpy as np
import matplotlib.pyplot as plt
import csv

class test_data:
  pass

def get_data():
  qrfname = "quadrect_pse_error.csv"
  thfname = "trihex_pse_error_tables.csv"

  class sdata:
    pass
  qr_dx = []
  qr_interp_l1 = []
  qr_interp_l1_rate = []
  qr_interp_l2 = []
  qr_interp_l2_rate = []
  qr_interp_linf = []
  qr_interp_linf_rate = []
  qr_lap_l1 = []
  qr_lap_l1_rate = []
  qr_lap_l2 = []
  qr_lap_l2_rate = []
  qr_lap_linf = []
  qr_lap_linf_rate = []
  with open(qrfname, 'r', encoding='utf-8-sig') as qrf:
    qrcsv = csv.DictReader(qrf, dialect='excel')
    for row in qrcsv:
      qr_dx.append(float(row['dx']))
      qr_interp_l1.append(float(row['interp_l1']))
      qr_interp_l2.append(float(row['interp_l2']))
      qr_interp_linf.append(float(row['interp_linf']))
      qr_lap_l1.append(float(row['lap_l1']))
      qr_lap_l2.append(float(row['lap_l2']))
      qr_lap_linf.append(float(row['lap_linf']))
      if row['interp_l1_rate'] == '---':
        qr_interp_l1_rate.append(np.nan)
        qr_interp_l2_rate.append(np.nan)
        qr_interp_linf_rate.append(np.nan)
        qr_lap_l1_rate.append(np.nan)
        qr_lap_l2_rate.append(np.nan)
        qr_lap_linf_rate.append(np.nan)
      else:
        qr_interp_l1_rate.append(float(row['interp_l1_rate']))
        qr_interp_l2_rate.append(float(row['interp_l2_rate']))
        qr_interp_linf_rate.append(float(row['interp_linf_rate']))
        qr_lap_l1_rate.append(float(row['lap_l1_rate']))
        qr_lap_l2_rate.append(float(row['lap_l2_rate']))
        qr_lap_linf_rate.append(float(row['lap_linf_rate']))
  sdata.qr_dx = np.array(qr_dx)
  sdata.qr_interp_l1 = np.array(qr_interp_l1)
  sdata.qr_interp_l1_rate = np.array(qr_interp_l1_rate)
  sdata.qr_interp_l2 = np.array(qr_interp_l2)
  sdata.qr_interp_l2_rate = np.array(qr_interp_l2_rate)
  sdata.qr_interp_linf = np.array(qr_interp_linf)
  sdata.qr_interp_linf_rate = np.array(qr_interp_linf_rate)
  sdata.qr_lap_l1 = np.array(qr_lap_l1)
  sdata.qr_lap_l1_rate = np.array(qr_lap_l1_rate)
  sdata.qr_lap_l2 = np.array(qr_lap_l2)
  sdata.qr_lap_l2_rate = np.array(qr_lap_l2_rate)
  sdata.qr_lap_linf = np.array(qr_lap_linf)
  sdata.qr_lap_linf_rate = np.array(qr_lap_linf_rate)

  th_dx = []
  th_interp_l1 = []
  th_interp_l1_rate = []
  th_interp_l2 = []
  th_interp_l2_rate = []
  th_interp_linf = []
  th_interp_linf_rate = []
  th_lap_l1 = []
  th_lap_l1_rate = []
  th_lap_l2 = []
  th_lap_l2_rate = []
  th_lap_linf = []
  th_lap_linf_rate = []
  with open(thfname, 'r', encoding='utf-8-sig') as thf:
    thcsv = csv.DictReader(thf, dialect='excel')
    for row in thcsv:
      th_dx.append(float(row['dx']))
      th_interp_l1.append(float(row['interp_l1']))
      th_interp_l2.append(float(row['interp_l2']))
      th_interp_linf.append(float(row['interp_linf']))
      th_lap_l1.append(float(row['lap_l1']))
      th_lap_l2.append(float(row['lap_l2']))
      th_lap_linf.append(float(row['lap_linf']))
      if row['interp_l1_rate'] == '---':
        th_interp_l1_rate.append(np.nan)
        th_interp_l2_rate.append(np.nan)
        th_interp_linf_rate.append(np.nan)
        th_lap_l1_rate.append(np.nan)
        th_lap_l2_rate.append(np.nan)
        th_lap_linf_rate.append(np.nan)
      else:
        th_interp_l1_rate.append(float(row['interp_l1_rate']))
        th_interp_l2_rate.append(float(row['interp_l2_rate']))
        th_interp_linf_rate.append(float(row['interp_linf_rate']))
        th_lap_l1_rate.append(float(row['lap_l1_rate']))
        th_lap_l2_rate.append(float(row['lap_l2_rate']))
        th_lap_linf_rate.append(float(row['lap_linf_rate']))
  sdata.th_dx = np.array(th_dx)
  sdata.th_interp_l1 = np.array(th_interp_l1)
  sdata.th_interp_l1_rate = np.array(th_interp_l1_rate)
  sdata.th_interp_l2 = np.array(th_interp_l2)
  sdata.th_interp_l2_rate = np.array(th_interp_l2_rate)
  sdata.th_interp_linf = np.array(th_interp_linf)
  sdata.th_interp_linf_rate = np.array(th_interp_linf_rate)
  sdata.th_lap_l1 = np.array(th_lap_l1)
  sdata.th_lap_l1_rate = np.array(th_lap_l1_rate)
  sdata.th_lap_l2 = np.array(th_lap_l2)
  sdata.th_lap_l2_rate = np.array(th_lap_l2_rate)
  sdata.th_lap_linf = np.array(th_lap_linf)
  sdata.th_lap_linf_rate = np.array(th_lap_linf_rate)
  return sdata


def plot_err(sdata):
  fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)

  cdx = 0.25*np.logspace(-1, 0)
  order8_ref = 500*np.power(cdx,8)

  ax0.loglog(sdata.qr_dx, sdata.qr_interp_l1, ':D', label='qr_l1')
  ax0.loglog(sdata.qr_dx, sdata.qr_interp_l2, '-d', label='qr_l2')
  ax0.loglog(sdata.qr_dx, sdata.qr_interp_linf, '--s', label='qr_linf')

  ax0.loglog(sdata.th_dx, sdata.th_interp_l1, ':v', label='th_l1')
  ax0.loglog(sdata.th_dx, sdata.th_interp_l2, '-^', label='th_l2')
  ax0.loglog(sdata.th_dx, sdata.th_interp_linf, '--<', label='th_linf')

  ax0.loglog(cdx, order8_ref, 'k:')

  ax1.loglog(sdata.qr_dx, sdata.qr_lap_l1, ':D', label='qr_l1')
  ax1.loglog(sdata.qr_dx, sdata.qr_lap_l2, '-d', label='qr_l2')
  ax1.loglog(sdata.qr_dx, sdata.qr_lap_linf, '--s', label='qr_linf')

  ax1.loglog(sdata.th_dx, sdata.th_lap_l1, ':v', label='th_l1')
  ax1.loglog(sdata.th_dx, sdata.th_lap_l2, '-^', label='th_l2')
  ax1.loglog(sdata.th_dx, sdata.th_lap_linf, '--<', label='th_linf')

  ax1.loglog(cdx, 10*order8_ref, 'k:', label='8th ord.')

  ax0.set(xlabel='dx', ylabel='rel. err.', title='interpolate Gaussian')
  ax1.set(xlabel='dx', title='Laplacian(Gaussian)')
  ax0.grid()
  ax1.grid()
  ax1.legend()

  fig.savefig('pse_err_plot.png', bbox_inches='tight')
  plt.close(fig)

if __name__ == '__main__':

  sdata = get_data()

  plot_err(sdata)
