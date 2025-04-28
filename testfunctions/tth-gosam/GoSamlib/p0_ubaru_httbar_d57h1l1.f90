module     p0_ubaru_httbar_d57h1l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity1d57h1l1.f90
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
      use p0_ubaru_httbar_abbrevd57h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc57(31)
      complex(ki) :: QspQ
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspl3
      QspQ = dotproduct(Q,Q)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspk2 = dotproduct(Q,k2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspl3 = dotproduct(Q,l3)
      acc57(1)=abb57(9)
      acc57(2)=abb57(10)
      acc57(3)=abb57(11)
      acc57(4)=abb57(12)
      acc57(5)=abb57(13)
      acc57(6)=abb57(14)
      acc57(7)=abb57(15)
      acc57(8)=abb57(16)
      acc57(9)=abb57(17)
      acc57(10)=abb57(18)
      acc57(11)=abb57(19)
      acc57(12)=abb57(20)
      acc57(13)=abb57(21)
      acc57(14)=abb57(22)
      acc57(15)=abb57(23)
      acc57(16)=abb57(25)
      acc57(17)=abb57(26)
      acc57(18)=abb57(28)
      acc57(19)=abb57(29)
      acc57(20)=abb57(30)
      acc57(21)=acc57(5)*QspQ
      acc57(22)=acc57(15)*Qspval4k2
      acc57(23)=acc57(17)*Qspval5l3
      acc57(24)=acc57(19)*Qspval5k2
      acc57(25)=acc57(20)*Qspval4l3
      acc57(21)=acc57(25)+acc57(24)+acc57(23)+acc57(22)+acc57(21)+acc57(3)
      acc57(21)=Qspvak1k2*acc57(21)
      acc57(22)=acc57(9)*Qspval5k2
      acc57(23)=acc57(13)*Qspval4k2
      acc57(22)=acc57(23)+acc57(2)+acc57(22)
      acc57(22)=QspQ*acc57(22)
      acc57(23)=acc57(9)*Qspval5l3
      acc57(23)=acc57(4)+acc57(23)
      acc57(23)=Qspval3k2*acc57(23)
      acc57(24)=acc57(12)*Qspval5k2
      acc57(24)=acc57(24)+acc57(11)
      acc57(24)=Qspk2*acc57(24)
      acc57(25)=acc57(6)*Qspval4l3
      acc57(26)=acc57(7)*Qspval4k2
      acc57(27)=acc57(8)*Qspval5l3
      acc57(28)=acc57(10)*Qspval5k2
      acc57(29)=Qspval3l5*acc57(18)
      acc57(30)=Qspval3k1*acc57(14)
      acc57(31)=Qspl3*acc57(16)
      brack=acc57(1)+acc57(21)+acc57(22)+acc57(23)+acc57(24)+acc57(25)+acc57(26&
      &)+acc57(27)+acc57(28)+acc57(29)+acc57(30)+acc57(31)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d57h1l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd57h1
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d57
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d57 = 0.0_ki
      d57 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d57, ki), aimag(d57), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d57h1l1
