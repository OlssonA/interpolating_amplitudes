module     p2_gg_httbar_d262h12l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d262h12l1.f90
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
      use p2_gg_httbar_abbrevd262h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc262(105)
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspe2
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: QspQ
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspk2
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval3l5
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspe2 = dotproduct(Q,e2)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      QspQ = dotproduct(Q,Q)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspk2 = dotproduct(Q,k2)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval3l5 = dotproduct(Q,spval3l5)
      acc262(1)=abb262(8)
      acc262(2)=abb262(9)
      acc262(3)=abb262(10)
      acc262(4)=abb262(11)
      acc262(5)=abb262(12)
      acc262(6)=abb262(13)
      acc262(7)=abb262(14)
      acc262(8)=abb262(15)
      acc262(9)=abb262(16)
      acc262(10)=abb262(17)
      acc262(11)=abb262(18)
      acc262(12)=abb262(19)
      acc262(13)=abb262(20)
      acc262(14)=abb262(21)
      acc262(15)=abb262(22)
      acc262(16)=abb262(23)
      acc262(17)=abb262(24)
      acc262(18)=abb262(25)
      acc262(19)=abb262(26)
      acc262(20)=abb262(27)
      acc262(21)=abb262(28)
      acc262(22)=abb262(29)
      acc262(23)=abb262(30)
      acc262(24)=abb262(31)
      acc262(25)=abb262(32)
      acc262(26)=abb262(33)
      acc262(27)=abb262(34)
      acc262(28)=abb262(35)
      acc262(29)=abb262(36)
      acc262(30)=abb262(37)
      acc262(31)=abb262(38)
      acc262(32)=abb262(39)
      acc262(33)=abb262(40)
      acc262(34)=abb262(41)
      acc262(35)=abb262(42)
      acc262(36)=abb262(43)
      acc262(37)=abb262(44)
      acc262(38)=abb262(45)
      acc262(39)=abb262(46)
      acc262(40)=abb262(47)
      acc262(41)=abb262(48)
      acc262(42)=abb262(49)
      acc262(43)=abb262(50)
      acc262(44)=abb262(51)
      acc262(45)=abb262(52)
      acc262(46)=abb262(53)
      acc262(47)=abb262(54)
      acc262(48)=abb262(55)
      acc262(49)=abb262(56)
      acc262(50)=abb262(57)
      acc262(51)=abb262(58)
      acc262(52)=abb262(59)
      acc262(53)=abb262(60)
      acc262(54)=abb262(61)
      acc262(55)=abb262(62)
      acc262(56)=abb262(63)
      acc262(57)=abb262(64)
      acc262(58)=abb262(65)
      acc262(59)=abb262(66)
      acc262(60)=abb262(67)
      acc262(61)=abb262(68)
      acc262(62)=abb262(69)
      acc262(63)=abb262(70)
      acc262(64)=abb262(71)
      acc262(65)=abb262(72)
      acc262(66)=abb262(73)
      acc262(67)=abb262(74)
      acc262(68)=abb262(75)
      acc262(69)=abb262(76)
      acc262(70)=abb262(78)
      acc262(71)=abb262(79)
      acc262(72)=abb262(81)
      acc262(73)=abb262(84)
      acc262(74)=abb262(85)
      acc262(75)=abb262(88)
      acc262(76)=abb262(91)
      acc262(77)=abb262(92)
      acc262(78)=abb262(96)
      acc262(79)=abb262(97)
      acc262(80)=abb262(106)
      acc262(81)=abb262(108)
      acc262(82)=abb262(109)
      acc262(83)=abb262(114)
      acc262(84)=abb262(116)
      acc262(85)=abb262(118)
      acc262(86)=Qspvae1l5*acc262(74)
      acc262(87)=-Qspvae1l4*acc262(79)
      acc262(86)=acc262(87)+acc262(52)+acc262(86)
      acc262(86)=Qspe2*acc262(86)
      acc262(87)=Qspvak2k1*acc262(50)
      acc262(88)=Qspvak1l5*acc262(67)
      acc262(89)=Qspval4e1*acc262(84)
      acc262(90)=Qspvae1l5*acc262(71)
      acc262(91)=Qspvae1l4*acc262(36)
      acc262(92)=Qspval3e1*acc262(61)
      acc262(93)=Qspvak2e1*acc262(38)
      acc262(94)=Qspvae1k2*acc262(32)
      acc262(95)=Qspvae1l3*acc262(82)
      acc262(96)=-Qspvak2e1*acc262(19)
      acc262(96)=acc262(63)+acc262(96)
      acc262(96)=Qspvae1e2*acc262(96)
      acc262(97)=Qspvae1l4*acc262(81)
      acc262(97)=acc262(1)+acc262(97)
      acc262(97)=Qspvae2e1*acc262(97)
      acc262(98)=QspQ*acc262(22)
      acc262(86)=acc262(98)+acc262(86)+acc262(97)+acc262(96)+acc262(95)+acc262(&
      &94)+acc262(93)+acc262(92)+acc262(91)+acc262(90)+acc262(89)+acc262(88)+ac&
      &c262(2)+acc262(87)
      acc262(86)=QspQ*acc262(86)
      acc262(87)=Qspvak1e1*acc262(45)
      acc262(88)=Qspval5e1*acc262(78)
      acc262(89)=Qspvak1l5*acc262(69)
      acc262(90)=Qspvak2l4*acc262(56)
      acc262(91)=Qspval3l4*acc262(57)
      acc262(92)=Qspval4e1*acc262(85)
      acc262(93)=Qspvak2l3*acc262(19)
      acc262(93)=acc262(58)+acc262(93)
      acc262(93)=Qspval3e1*acc262(93)
      acc262(94)=Qspk2*acc262(11)
      acc262(94)=acc262(4)+acc262(94)
      acc262(94)=Qspvak2e1*acc262(94)
      acc262(87)=acc262(94)+acc262(93)+acc262(92)+acc262(91)+acc262(90)+acc262(&
      &89)+acc262(88)+acc262(8)+acc262(87)
      acc262(87)=Qspvae1e2*acc262(87)
      acc262(88)=Qspvae1k1*acc262(40)
      acc262(89)=Qspk2*acc262(59)
      acc262(90)=Qspvak2k1*acc262(42)
      acc262(91)=Qspvak2l3*acc262(39)
      acc262(92)=Qspvae1l5*acc262(75)
      acc262(93)=Qspvae1l4*acc262(77)
      acc262(94)=-Qspvak2l4*acc262(49)
      acc262(94)=acc262(12)+acc262(94)
      acc262(94)=Qspvae1k2*acc262(94)
      acc262(95)=-Qspval3l4*acc262(81)
      acc262(95)=acc262(7)+acc262(95)
      acc262(95)=Qspvae1l3*acc262(95)
      acc262(88)=acc262(95)+acc262(94)+acc262(93)+acc262(92)+acc262(91)+acc262(&
      &90)+acc262(89)+acc262(24)+acc262(88)
      acc262(88)=Qspvae2e1*acc262(88)
      acc262(89)=Qspvak2l5*acc262(64)
      acc262(90)=Qspvak1l5*acc262(21)
      acc262(91)=Qspvak2l4*acc262(13)
      acc262(92)=Qspval4e1*acc262(68)
      acc262(93)=Qspval3e1*acc262(70)
      acc262(94)=Qspvak2e1*acc262(16)
      acc262(89)=acc262(94)+acc262(93)+acc262(92)+acc262(91)+acc262(90)+acc262(&
      &14)+acc262(89)
      acc262(89)=Qspvae1k2*acc262(89)
      acc262(90)=Qspval3l5*acc262(37)
      acc262(91)=Qspvak1l5*acc262(53)
      acc262(92)=Qspval3l4*acc262(34)
      acc262(93)=Qspval4e1*acc262(72)
      acc262(94)=Qspval3e1*acc262(48)
      acc262(95)=Qspvak2e1*acc262(15)
      acc262(90)=acc262(95)+acc262(94)+acc262(93)+acc262(92)+acc262(91)+acc262(&
      &3)+acc262(90)
      acc262(90)=Qspvae1l3*acc262(90)
      acc262(91)=Qspvae1l5*acc262(26)
      acc262(92)=Qspvae1l4*acc262(28)
      acc262(91)=acc262(92)+acc262(20)+acc262(91)
      acc262(91)=Qspvak2e1*acc262(91)
      acc262(92)=Qspvak2l5*acc262(65)
      acc262(93)=Qspvak2l4*acc262(17)
      acc262(92)=acc262(93)+acc262(5)+acc262(92)
      acc262(92)=Qspvae1k2*acc262(92)
      acc262(93)=-acc262(74)*Qspval3l5
      acc262(94)=Qspval3l4*acc262(79)
      acc262(93)=acc262(94)+acc262(6)+acc262(93)
      acc262(93)=Qspvae1l3*acc262(93)
      acc262(94)=Qspvae1l5*acc262(35)
      acc262(95)=Qspvae1l4*acc262(30)
      acc262(96)=Qspval3e1*acc262(44)
      acc262(91)=acc262(93)+acc262(92)+acc262(91)+acc262(96)+acc262(95)+acc262(&
      &55)+acc262(94)
      acc262(91)=Qspe2*acc262(91)
      acc262(92)=Qspvak2k1*acc262(51)
      acc262(93)=Qspvak2l3*acc262(33)
      acc262(94)=Qspvae1l5*acc262(73)
      acc262(95)=Qspvae1l4*acc262(62)
      acc262(92)=acc262(95)+acc262(94)+acc262(93)+acc262(23)+acc262(92)
      acc262(92)=Qspval3e1*acc262(92)
      acc262(93)=Qspk2*acc262(60)
      acc262(94)=Qspvae1l5*acc262(25)
      acc262(95)=Qspvae1l4*acc262(29)
      acc262(93)=acc262(95)+acc262(94)+acc262(10)+acc262(93)
      acc262(93)=Qspvak2e1*acc262(93)
      acc262(94)=Qspvak1e1*acc262(43)
      acc262(95)=Qspvae1k1*acc262(27)
      acc262(96)=Qspval5e1*acc262(76)
      acc262(97)=Qspk2*acc262(80)
      acc262(98)=Qspvak2k1*acc262(41)
      acc262(99)=Qspvak2l3*acc262(18)
      acc262(100)=Qspvak1l5*acc262(66)
      acc262(101)=Qspvak2l4*acc262(46)
      acc262(102)=Qspval3l4*acc262(47)
      acc262(103)=Qspval4e1*acc262(83)
      acc262(104)=Qspvae1l5*acc262(31)
      acc262(105)=Qspvae1l4*acc262(54)
      brack=acc262(9)+acc262(86)+acc262(87)+acc262(88)+acc262(89)+acc262(90)+ac&
      &c262(91)+acc262(92)+acc262(93)+acc262(94)+acc262(95)+acc262(96)+acc262(9&
      &7)+acc262(98)+acc262(99)+acc262(100)+acc262(101)+acc262(102)+acc262(103)&
      &+acc262(104)+acc262(105)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d262h12l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd262h12
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d262
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k3-k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d262 = 0.0_ki
      d262 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d262, ki), aimag(d262), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d262h12l1
