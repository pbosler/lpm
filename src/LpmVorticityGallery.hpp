#ifndef LPM_VORTICITY_GALLERY_HPP
#define LPM_VORTICITY_GALLERY_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "Kokkos_Core.hpp"
#include <memory>

namespace Lpm {

struct VorticityInitialCondition {
  typedef std::shared_ptr<VorticityInitialCondition> ptr;

  virtual ~VorticityInitialCondition() {}

  virtual Real eval(const Real& x, const Real& y, const Real& z) const = 0;

  virtual std::string name() const {return std::string();}
};

struct SolidBodyRotation : public VorticityInitialCondition {
  static constexpr Real OMEGA = 2*PI;

  inline Real eval(const Real& x, const Real& y, const Real& z) const {return 2*OMEGA*z;}

  inline std::string name() const override {return "rotation";}
};

}
#endif
