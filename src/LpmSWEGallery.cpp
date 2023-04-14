#include "LpmSWEGallery.hpp"

namespace Lpm {

// Real Thacker81FlatOscillation::eval_sfc(const Real& x, const Real& y, const
// Real& z) const {
//   return 2*eta*D0/L*(x/L - eta/(2*L));
//
// }
//
// Real Thacker81FlatOscillation::eval_h(const Real& x, const Real& y, const
// Real& z) const {
//   return eval_sfc(x,y,z) - Thacker81BasinTopo::eval(x,y);
// }
//
// Real Thacker81CurvedOscillation::eval_sfc(const Real& x, const Real& y, const
// Real& z) const {
//   return D0*(std::sqrt(1-square(Real(A)))/(1-Real(A)) - 1 -
//   (square(x)+square(y))/square(Real(L))*
//     ((1-square(Real(A)))/square(1-Real(A)) -1 ));
// }
//
// Real Thacker81CurvedOscillation::eval_h(const Real& x, const Real& y, const
// Real& z) const {
//   return eval_sfc(x,y,z) - Thacker81BasinTopo::eval(x,y);
// }
//
// Real Thacker81CurvedOscillation::exact_sfc(const Real& x, const Real& y,
// const Real &t) const {
//   return D0*(std::sqrt(1-square(Real(A)))/(1-Real(A)*std::cos(omg()*t)) - 1 -
//     (square(x)+square(y))/square(Real(L))*
//   ((1-square(Real(A)))/square(1-Real(A)*std::cos(omg()*t))-1));
// }
//
// void Thacker81CurvedOscillation::exact_velocity(Real& u, Real& v, const Real&
// x, const Real& y,
//   const Real& t) const {
//   const Real cosomgt = std::cos(omg()*t);
//   const Real sinomgt = std::sin(omg()*t);
//   const Real prefac = 1/(1-A*cosomgt);
//   const Real rfac = std::sqrt(1-square(Real(A)));
//   u = prefac * (0.5*omg()*x*Real(A)*sinomgt -
//   0.5*f()*y*(rfac+Real(A)*cosomgt-1)); v = prefac *
//   (0.5*omg()*y*Real(A)*sinomgt + 0.5*f()*x*(rfac+Real(A)*cosomgt-1));
// }

}
