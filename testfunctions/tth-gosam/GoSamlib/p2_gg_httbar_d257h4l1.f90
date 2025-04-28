module     p2_gg_httbar_d257h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d257h4l1.f90
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
      use p2_gg_httbar_abbrevd257h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc257(78)
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae2l4
      complex(ki) :: QspQ
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae2k1
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspk2 = dotproduct(Q,k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      QspQ = dotproduct(Q,Q)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      acc257(1)=abb257(7)
      acc257(2)=abb257(8)
      acc257(3)=abb257(9)
      acc257(4)=abb257(10)
      acc257(5)=abb257(11)
      acc257(6)=abb257(12)
      acc257(7)=abb257(13)
      acc257(8)=abb257(14)
      acc257(9)=abb257(15)
      acc257(10)=abb257(16)
      acc257(11)=abb257(17)
      acc257(12)=abb257(18)
      acc257(13)=abb257(19)
      acc257(14)=abb257(20)
      acc257(15)=abb257(21)
      acc257(16)=abb257(22)
      acc257(17)=abb257(23)
      acc257(18)=abb257(24)
      acc257(19)=abb257(25)
      acc257(20)=abb257(26)
      acc257(21)=abb257(27)
      acc257(22)=abb257(28)
      acc257(23)=abb257(29)
      acc257(24)=abb257(30)
      acc257(25)=abb257(31)
      acc257(26)=abb257(32)
      acc257(27)=abb257(33)
      acc257(28)=abb257(34)
      acc257(29)=abb257(35)
      acc257(30)=abb257(36)
      acc257(31)=abb257(37)
      acc257(32)=abb257(38)
      acc257(33)=abb257(39)
      acc257(34)=abb257(40)
      acc257(35)=abb257(41)
      acc257(36)=abb257(42)
      acc257(37)=abb257(43)
      acc257(38)=abb257(44)
      acc257(39)=abb257(45)
      acc257(40)=abb257(46)
      acc257(41)=abb257(47)
      acc257(42)=abb257(48)
      acc257(43)=abb257(49)
      acc257(44)=abb257(50)
      acc257(45)=abb257(51)
      acc257(46)=abb257(52)
      acc257(47)=abb257(53)
      acc257(48)=abb257(54)
      acc257(49)=abb257(55)
      acc257(50)=abb257(56)
      acc257(51)=abb257(57)
      acc257(52)=abb257(58)
      acc257(53)=abb257(59)
      acc257(54)=abb257(62)
      acc257(55)=abb257(65)
      acc257(56)=abb257(66)
      acc257(57)=abb257(67)
      acc257(58)=-Qspvak2l4*acc257(18)
      acc257(59)=Qspk2*acc257(22)
      acc257(60)=-Qspvak2e1*acc257(46)
      acc257(58)=acc257(60)+acc257(59)+acc257(4)+acc257(58)
      acc257(58)=Qspvae2k2*acc257(58)
      acc257(59)=Qspval3k2*acc257(1)
      acc257(60)=-Qspval3l4*acc257(35)
      acc257(59)=acc257(60)+acc257(48)+acc257(59)
      acc257(59)=Qspvae2l3*acc257(59)
      acc257(60)=Qspvae2l4*acc257(35)
      acc257(61)=-Qspvae2k2*acc257(1)
      acc257(60)=acc257(61)+acc257(45)+acc257(60)
      acc257(60)=QspQ*acc257(60)
      acc257(61)=acc257(42)*Qspvak1k2
      acc257(62)=Qspval3k2*acc257(52)
      acc257(63)=Qspval3l4*acc257(31)
      acc257(64)=Qspvae2l4*acc257(53)
      acc257(65)=Qspval5k2*acc257(44)
      acc257(66)=Qspvak2e1*acc257(51)
      acc257(58)=acc257(60)+acc257(58)+acc257(66)+acc257(59)+acc257(65)+acc257(&
      &64)+acc257(63)+acc257(62)+acc257(61)+acc257(38)
      acc257(58)=Qspvae1e2*acc257(58)
      acc257(59)=Qspval5k2*acc257(34)
      acc257(60)=Qspk2*acc257(26)
      acc257(59)=acc257(60)+acc257(41)+acc257(59)
      acc257(59)=Qspvak2e2*acc257(59)
      acc257(60)=-Qspvak2l3*acc257(9)
      acc257(61)=Qspval5l3*acc257(5)
      acc257(60)=acc257(61)+acc257(16)+acc257(60)
      acc257(60)=Qspval3e2*acc257(60)
      acc257(61)=-Qspval5e2*acc257(5)
      acc257(62)=Qspvak2e2*acc257(9)
      acc257(61)=acc257(62)+acc257(14)+acc257(61)
      acc257(61)=QspQ*acc257(61)
      acc257(62)=-acc257(33)*Qspvak2k1
      acc257(63)=Qspvak2l3*acc257(50)
      acc257(64)=Qspval5l3*acc257(36)
      acc257(65)=Qspval5e2*acc257(6)
      acc257(66)=Qspk2*acc257(17)
      acc257(67)=Qspval5e2*acc257(56)
      acc257(67)=acc257(27)+acc257(67)
      acc257(67)=Qspvae1l4*acc257(67)
      acc257(59)=acc257(61)+acc257(67)+acc257(60)+acc257(59)+acc257(66)+acc257(&
      &65)+acc257(64)+acc257(63)+acc257(62)+acc257(11)
      acc257(59)=Qspvae2e1*acc257(59)
      acc257(60)=Qspval5k2*acc257(2)
      acc257(61)=-Qspvae2l3*acc257(57)
      acc257(62)=-Qspvae2k2*acc257(28)
      acc257(60)=acc257(62)+acc257(61)+acc257(13)+acc257(60)
      acc257(60)=Qspvae1l4*acc257(60)
      acc257(61)=Qspk2*acc257(49)
      acc257(62)=Qspval3e2*acc257(23)
      acc257(61)=acc257(62)+acc257(15)+acc257(61)
      acc257(61)=Qspvak2e1*acc257(61)
      acc257(62)=Qspvak2e1*acc257(47)
      acc257(63)=Qspvae1l4*acc257(37)
      acc257(62)=acc257(63)+acc257(20)+acc257(62)
      acc257(62)=QspQ*acc257(62)
      acc257(63)=acc257(43)*Qspvae1k2
      acc257(64)=acc257(32)*Qspvak1e2
      acc257(65)=acc257(8)*Qspvae2k1
      acc257(66)=Qspvak2l4*acc257(25)
      acc257(67)=Qspvak2l3*acc257(10)
      acc257(68)=Qspval3k2*acc257(30)
      acc257(69)=Qspval3l4*acc257(19)
      acc257(70)=Qspval5l3*acc257(7)
      acc257(71)=Qspvae2l4*acc257(55)
      acc257(72)=Qspval5k2*acc257(40)
      acc257(73)=Qspval5e2*acc257(39)
      acc257(74)=Qspk2*acc257(21)
      acc257(75)=Qspvak2e2*acc257(24)
      acc257(76)=Qspval3e2*acc257(54)
      acc257(77)=Qspvae2l3*acc257(29)
      acc257(78)=Qspvae2k2*acc257(3)
      brack=acc257(12)+acc257(58)+acc257(59)+acc257(60)+acc257(61)+acc257(62)+a&
      &cc257(63)+acc257(64)+acc257(65)+acc257(66)+acc257(67)+acc257(68)+acc257(&
      &69)+acc257(70)+acc257(71)+acc257(72)+acc257(73)+acc257(74)+acc257(75)+ac&
      &c257(76)+acc257(77)+acc257(78)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d257h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd257h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d257
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k3-k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d257 = 0.0_ki
      d257 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d257, ki), aimag(d257), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d257h4l1
