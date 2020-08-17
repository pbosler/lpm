#ifndef LPM_VORTICITY_GALLERY_HPP
#define LPM_VORTICITY_GALLERY_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "Kokkos_Core.hpp"
#include <memory>
#include <cmath>

namespace Lpm {

struct VorticityInitialCondition {
  typedef std::shared_ptr<VorticityInitialCondition> ptr;

  virtual ~VorticityInitialCondition() {}

  virtual Real eval(const Real& x, const Real& y, const Real& z) const = 0;

  virtual Real eval(const Real& x, const Real& y) const = 0;

  virtual std::string name() const {return std::string();}
};

struct SolidBodyRotation : public VorticityInitialCondition {
  static constexpr Real OMEGA = 2*PI;

  inline Real eval(const Real& x, const Real& y, const Real& z) const {return 2*OMEGA*z;}

  inline Real eval(const Real& x, const Real& y) const {return 0;}

  inline std::string name() const override {return "rotation";}

  inline void init_velocity(Real& u, Real& v, Real& w,
    const Real& x, const Real& y, const Real& z) const {
    u = -OMEGA*y;
    v =  OMEGA*x;
    w = 0;
  }
};

struct NitscheStricklandVortex : public VorticityInitialCondition {
  static constexpr Real b = 0.5;

  inline Real eval(const Real& x, const Real& y, const Real& z) const {return 0;}

  inline Real eval(const Real& x, const Real& y) const {
    const Real rsq = square(x) + square(y);
    const Real r = sqrt(rsq);
    return (3*safe_divide(r) - 2*b*r)*rsq*std::exp(-b*rsq);
  }

  inline std::string name() const {return "Nitsche&Strickland";}

  inline void init_velocity(Real& u, Real& v, const Real& x, const Real& y) const {
    const Real rsq = square(x) + square(y);
    const Real utheta = rsq*std::exp(-b*rsq);
    const Real theta = std::atan2(y,x);
    u = -utheta*std::sin(theta);
    v =  utheta*std::cos(theta);
  }
};


}
#endif
