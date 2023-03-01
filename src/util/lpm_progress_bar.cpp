#include "lpm_progress_bar.hpp"

namespace Lpm {

ProgressBar::ProgressBar(const std::string& name, const Int niterations,
                         const Real write_freq, std::ostream& os)
    : name_(name),
      niter_(niterations),
      freq_(write_freq),
      it_(0),
      next_(0),
      os_(os) {
  os_ << name_ << ":";
  os_.flush();
}

void ProgressBar::update() {
  ++it_;
  const Real p = 100 * it_ / niter_;
  if (p >= next_ || it_ == niter_) {
    os_ << " " << p;
    if (it_ == niter_) {
      os_ << "\n";
    }
    os_.flush();
    next_ += freq_;
  }
}

}  // namespace Lpm
