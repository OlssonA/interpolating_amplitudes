module     p2_gg_httbar_d67h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d67h4l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd67h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc67(41)
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval5k1
      complex(ki) :: QspQ
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspvak2l3
      Qspk2 = dotproduct(Q,k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval5k1 = dotproduct(Q,spval5k1)
      QspQ = dotproduct(Q,Q)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      acc67(1)=abb67(10)
      acc67(2)=abb67(11)
      acc67(3)=abb67(12)
      acc67(4)=abb67(13)
      acc67(5)=abb67(14)
      acc67(6)=abb67(15)
      acc67(7)=abb67(16)
      acc67(8)=abb67(17)
      acc67(9)=abb67(18)
      acc67(10)=abb67(19)
      acc67(11)=abb67(20)
      acc67(12)=abb67(21)
      acc67(13)=abb67(22)
      acc67(14)=abb67(23)
      acc67(15)=abb67(24)
      acc67(16)=abb67(25)
      acc67(17)=abb67(26)
      acc67(18)=abb67(28)
      acc67(19)=abb67(30)
      acc67(20)=abb67(34)
      acc67(21)=abb67(35)
      acc67(22)=abb67(44)
      acc67(23)=abb67(45)
      acc67(24)=abb67(48)
      acc67(25)=abb67(49)
      acc67(26)=Qspk2*acc67(20)
      acc67(27)=Qspvak1k2*acc67(14)
      acc67(28)=Qspvak1l4*acc67(19)
      acc67(29)=Qspvak2k1*acc67(4)
      acc67(30)=Qspval5k1*acc67(22)
      acc67(26)=acc67(30)+acc67(29)+acc67(28)+acc67(27)+acc67(1)+acc67(26)
      acc67(26)=QspQ*acc67(26)
      acc67(27)=acc67(18)*Qspvak2l4
      acc67(27)=acc67(27)+acc67(21)
      acc67(27)=Qspval5k2*acc67(27)
      acc67(28)=acc67(23)*Qspval5l3
      acc67(29)=acc67(17)*Qspval3l5
      acc67(30)=acc67(13)*Qspval3k2
      acc67(31)=acc67(11)*Qspval3l4
      acc67(32)=acc67(9)*Qspval4k2
      acc67(33)=acc67(8)*Qspvak2l5
      acc67(34)=acc67(5)*Qspval4l3
      acc67(35)=acc67(2)*Qspvak2l3
      acc67(36)=Qspvak2l4*acc67(7)
      acc67(37)=Qspk2*acc67(25)
      acc67(37)=acc67(3)+acc67(37)
      acc67(37)=Qspk2*acc67(37)
      acc67(38)=Qspvak1k2*acc67(10)
      acc67(39)=Qspvak1l4*acc67(12)
      acc67(40)=Qspvak1k2*acc67(24)
      acc67(40)=acc67(16)+acc67(40)
      acc67(40)=Qspvak2k1*acc67(40)
      acc67(41)=-Qspvak1l4*acc67(18)
      acc67(41)=acc67(15)+acc67(41)
      acc67(41)=Qspval5k1*acc67(41)
      brack=acc67(6)+acc67(26)+acc67(27)+acc67(28)+acc67(29)+acc67(30)+acc67(31&
      &)+acc67(32)+acc67(33)+acc67(34)+acc67(35)+acc67(36)+acc67(37)+acc67(38)+&
      &acc67(39)+acc67(40)+acc67(41)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d67h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd67h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d67
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d67 = 0.0_ki
      d67 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d67, ki), aimag(d67), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d67h4l1
