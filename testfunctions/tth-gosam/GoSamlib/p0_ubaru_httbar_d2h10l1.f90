module     p0_ubaru_httbar_d2h10l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d2h10l1.f90
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
      use p0_ubaru_httbar_abbrevd2h10
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc2(21)
      complex(ki) :: Qspk2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2k1
      complex(ki) :: QspQ
      complex(ki) :: Qspl5
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspk1
      Qspk2 = dotproduct(Q,k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      QspQ = dotproduct(Q,Q)
      Qspl5 = dotproduct(Q,l5)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspk1 = dotproduct(Q,k1)
      acc2(1)=abb2(9)
      acc2(2)=abb2(10)
      acc2(3)=abb2(11)
      acc2(4)=abb2(12)
      acc2(5)=abb2(13)
      acc2(6)=abb2(14)
      acc2(7)=abb2(15)
      acc2(8)=abb2(17)
      acc2(9)=abb2(21)
      acc2(10)=abb2(23)
      acc2(11)=abb2(24)
      acc2(12)=abb2(28)
      acc2(13)=abb2(31)
      acc2(14)=acc2(2)*Qspk2
      acc2(15)=acc2(4)*Qspval4l5
      acc2(16)=acc2(7)*Qspval3l5
      acc2(17)=acc2(8)*Qspvak2l3
      acc2(14)=acc2(17)+acc2(16)+acc2(15)+acc2(1)+acc2(14)
      acc2(14)=Qspvak2k1*acc2(14)
      acc2(15)=QspQ+Qspl5
      acc2(15)=acc2(9)*acc2(15)
      acc2(16)=acc2(5)*Qspk2
      acc2(17)=acc2(6)*Qspvak2l3
      acc2(18)=acc2(12)*Qspval4l5
      acc2(19)=acc2(13)*Qspval3l5
      acc2(20)=Qspvak2l5*acc2(10)
      acc2(21)=Qspk1*acc2(11)
      brack=acc2(3)+acc2(14)+acc2(15)+acc2(16)+acc2(17)+acc2(18)+acc2(19)+acc2(&
      &20)+acc2(21)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d2h10l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd2h10
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
end module p0_ubaru_httbar_d2h10l1
