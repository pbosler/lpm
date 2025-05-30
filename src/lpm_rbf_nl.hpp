#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"
#include "KokkosBatched_Set_Decl.hpp"
#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBatched_ApplyPivot_Decl.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Trsv_Decl.hpp"
#include "KokkosBatched_QR_WithColumnPivoting_Decl.hpp"
#include "KokkosBatched_ApplyQ_Decl.hpp"


using namespace KokkosBatched;



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*Solver Class for QR with column pivoting*/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
  template<typename DeviceType,
           typename ViewMatrixType,
           typename ViewVectorType,
      	   typename PivotViewType,
           typename WorkViewType,
           typename AlgoTagType>
  struct Functor_SolverBatchedTeamVectorQR_WithColumnPivoting {
    ViewMatrixType _a;
    ViewVectorType _x, _b, _t;
    PivotViewType _p;
    WorkViewType _w;

    KOKKOS_INLINE_FUNCTION
    Functor_SolverBatchedTeamVectorQR_WithColumnPivoting(const ViewMatrixType &a,
						       const ViewVectorType &x,
						       const ViewVectorType &b,
						       const ViewVectorType &t,
						       const PivotViewType &p,
						       const WorkViewType &w)
      : _a(a), _x(x), _b(b), _t(t), _p(p), _w(w) {} 

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &member) const {
      typedef typename ViewMatrixType::non_const_value_type value_type;
      const value_type one(1), zero(0);

      const int k = member.league_rank();
      auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
      auto bb = Kokkos::subview(_b, k, Kokkos::ALL());
      auto xx = Kokkos::subview(_x, k, Kokkos::ALL());
      auto tt = Kokkos::subview(_t, k, Kokkos::ALL());
      auto pp = Kokkos::subview(_p, k, Kokkos::ALL());
      auto ww = Kokkos::subview(_w, k, Kokkos::ALL());

      /// AA P^T = QR
      int matrix_rank(0);
      TeamVectorQR_WithColumnPivoting<MemberType,AlgoTagType>::invoke(member, aa, tt, pp, ww, matrix_rank);
      member.team_barrier();

      /// xx = bb;
      TeamVectorCopy<MemberType,Trans::NoTranspose>::invoke(member, bb, xx);
      member.team_barrier();

      /// bb = Q^{T} bb;
      TeamVectorApplyQ<MemberType,Side::Left,Trans::Transpose,Algo::ApplyQ::Unblocked>::invoke(member, aa, tt, xx, ww);
      member.team_barrier();

      /// xx = R^{-1} xx;
      TeamVectorTrsv<MemberType,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsv::Unblocked>
        ::invoke(member, one, aa, xx);
      member.team_barrier();

      /// xx = P xx
      TeamVectorApplyPivot<MemberType,Side::Left,Direct::Backward>::invoke(member, pp, xx);
      member.team_barrier();
    }

    inline
    void run() {
      typedef typename ViewMatrixType::non_const_value_type value_type;

      const int league_size = _a.extent(0);
      Kokkos::TeamPolicy<DeviceType> policy(league_size, Kokkos::AUTO);

//      Kokkos::parallel_for(name.c_str(), policy, *this);
  //    Kokkos::Profiling::popRegion(); 
    }
  };

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*Testing Class for QR with column pivoting*/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*  template<typename DeviceType,
           typename ViewMatrixType,
           typename ViewVectorType,
	   typename PivotViewType,
           typename WorkViewType,
           typename AlgoTagType>
  struct Functor_TestBatchedTeamVectorQR_WithColumnPivoting {
    ViewMatrixType _a;
    ViewVectorType _x, _b, _t;
    PivotViewType _p;
    WorkViewType _w;

    KOKKOS_INLINE_FUNCTION
    Functor_TestBatchedTeamVectorQR_WithColumnPivoting(const ViewMatrixType &a,
						       const ViewVectorType &x,
						       const ViewVectorType &b,
						       const ViewVectorType &t,
						       const PivotViewType &p,
						       const WorkViewType &w)
      : _a(a), _x(x), _b(b), _t(t), _p(p), _w(w) {} 

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &member) const {
      typedef typename ViewMatrixType::non_const_value_type value_type;
      const value_type one(1), zero(0), add_this(10);

      const int k = member.league_rank();
      auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
      auto bb = Kokkos::subview(_b, k, Kokkos::ALL());
      auto xx = Kokkos::subview(_x, k, Kokkos::ALL());
      auto tt = Kokkos::subview(_t, k, Kokkos::ALL());
      auto pp = Kokkos::subview(_p, k, Kokkos::ALL());
      auto ww = Kokkos::subview(_w, k, Kokkos::ALL());

      // make diagonal dominant; xx = 1,2,3,4...
      const int m = aa.extent(0);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                           [&](const int &i) {
                             aa(i,i) += add_this;
                             xx(i) = i+1;			     
                           });
      member.team_barrier();

      /// bb = AA*xx
      TeamVectorGemv<MemberType,Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(member, one, aa, xx, zero, bb); 
      member.team_barrier();

      /// AA P^T = QR
      int matrix_rank(0);
      TeamVectorQR_WithColumnPivoting<MemberType,AlgoTagType>::invoke(member, aa, tt, pp, ww, matrix_rank);
      member.team_barrier();

      /// xx = bb;
      TeamVectorCopy<MemberType,Trans::NoTranspose>::invoke(member, bb, xx);
      member.team_barrier();

      /// xx = Q^{T} xx;
      TeamVectorApplyQ<MemberType,Side::Left,Trans::Transpose,Algo::ApplyQ::Unblocked>::invoke(member, aa, tt, xx, ww);
      member.team_barrier();

      /// xx = R^{-1} xx
      TeamVectorTrsv<MemberType,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsv::Unblocked>
        ::invoke(member, one, aa, xx);
      member.team_barrier();

      /// xx = P xx
      TeamVectorApplyPivot<MemberType,Side::Left,Direct::Backward>::invoke(member, pp, xx);
      member.team_barrier();
    }

    inline
    void run() {
      typedef typename ViewMatrixType::non_const_value_type value_type;
      std::string name_region("KokkosBatched::Test::TeamVectorQR_WithColumnPivoting");
      std::string name_value_type = ( std::is_same<value_type,float>::value ? "::Float" : 
                                      std::is_same<value_type,double>::value ? "::Double" :
                                      std::is_same<value_type,Kokkos::complex<float> >::value ? "::ComplexFloat" :
                                      std::is_same<value_type,Kokkos::complex<double> >::value ? "::ComplexDouble" : "::UnknownValueType" );                               
      std::string name = name_region + name_value_type;
      Kokkos::Profiling::pushRegion( name.c_str() );

      const int league_size = _a.extent(0);
      Kokkos::TeamPolicy<DeviceType> policy(league_size, Kokkos::AUTO);

      Kokkos::parallel_for(name.c_str(), policy, *this);
      Kokkos::Profiling::popRegion(); 
    }
  };

*/
