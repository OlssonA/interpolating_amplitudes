module     p0_ubaru_httbar_d4h6l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d4h6l1.f90
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
      use p0_ubaru_httbar_abbrevd4h6
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc4(21)
      complex(ki) :: Qspk2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2k1
      complex(ki) :: QspQ
      complex(ki) :: Qspl4
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspk1
      Qspk2 = dotproduct(Q,k2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      QspQ = dotproduct(Q,Q)
      Qspl4 = dotproduct(Q,l4)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspk1 = dotproduct(Q,k1)
      acc4(1)=abb4(9)
      acc4(2)=abb4(10)
      acc4(3)=abb4(11)
      acc4(4)=abb4(12)
      acc4(5)=abb4(13)
      acc4(6)=abb4(14)
      acc4(7)=abb4(15)
      acc4(8)=abb4(17)
      acc4(9)=abb4(21)
      acc4(10)=abb4(23)
      acc4(11)=abb4(24)
      acc4(12)=abb4(28)
      acc4(13)=abb4(31)
      acc4(14)=acc4(2)*Qspk2
      acc4(15)=acc4(4)*Qspval5l4
      acc4(16)=acc4(7)*Qspval3l4
      acc4(17)=acc4(8)*Qspvak2l3
      acc4(14)=acc4(17)+acc4(16)+acc4(15)+acc4(1)+acc4(14)
      acc4(14)=Qspvak2k1*acc4(14)
      acc4(15)=QspQ+Qspl4
      acc4(15)=acc4(9)*acc4(15)
      acc4(16)=acc4(5)*Qspk2
      acc4(17)=acc4(6)*Qspvak2l3
      acc4(18)=acc4(12)*Qspval5l4
      acc4(19)=acc4(13)*Qspval3l4
      acc4(20)=Qspvak2l4*acc4(10)
      acc4(21)=Qspk1*acc4(11)
      brack=acc4(3)+acc4(14)+acc4(15)+acc4(16)+acc4(17)+acc4(18)+acc4(19)+acc4(&
      &20)+acc4(21)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d4h6l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd4h6
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d4
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k4+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d4 = 0.0_ki
      d4 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d4, ki), aimag(d4), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d4h6l1
