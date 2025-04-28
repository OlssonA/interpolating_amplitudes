module     p0_ubaru_httbar_d2h1l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity1d2h1l1.f90
   ! generator: buildfortran.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd2h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc2(19)
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: QspQ
      complex(ki) :: Qspk2
      complex(ki) :: Qspl5
      complex(ki) :: Qspk1
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      QspQ = dotproduct(Q,Q)
      Qspk2 = dotproduct(Q,k2)
      Qspl5 = dotproduct(Q,l5)
      Qspk1 = dotproduct(Q,k1)
      acc2(1)=abb2(9)
      acc2(2)=abb2(10)
      acc2(3)=abb2(11)
      acc2(4)=abb2(12)
      acc2(5)=abb2(14)
      acc2(6)=abb2(15)
      acc2(7)=abb2(16)
      acc2(8)=abb2(17)
      acc2(9)=abb2(18)
      acc2(10)=abb2(22)
      acc2(11)=abb2(24)
      acc2(12)=abb2(25)
      acc2(13)=acc2(2)*Qspval5k2
      acc2(14)=acc2(4)*Qspval5l3
      acc2(15)=acc2(5)*Qspval3k2
      acc2(16)=-acc2(10)*Qspval4k2
      acc2(13)=acc2(16)+acc2(15)+acc2(14)+acc2(3)+acc2(13)
      acc2(13)=Qspvak1k2*acc2(13)
      acc2(14)=-QspQ+Qspk2-Qspl5
      acc2(14)=acc2(9)*acc2(14)
      acc2(15)=acc2(6)*Qspval5k2
      acc2(16)=acc2(7)*Qspval4k2
      acc2(17)=acc2(8)*Qspval3k2
      acc2(18)=acc2(12)*Qspval5l3
      acc2(19)=Qspk1*acc2(11)
      brack=acc2(1)+acc2(13)+acc2(14)+acc2(15)+acc2(16)+acc2(17)+acc2(18)+acc2(&
      &19)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d2h1l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd2h1
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d2
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(-Q_ext(0:3),  ki_nin), aimag(-Q_ext(0:3)), ki)
      d2 = 0.0_ki
      d2 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d2, ki), aimag(d2), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d2h1l1
