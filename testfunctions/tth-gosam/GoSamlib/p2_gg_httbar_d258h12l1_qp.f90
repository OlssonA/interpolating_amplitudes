module     p2_gg_httbar_d258h12l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d258h12l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd258h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc258(162)
      complex(ki) :: Qspl4
      complex(ki) :: Qspl5
      complex(ki) :: QspQ
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspe1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspk1
      complex(ki) :: Qspk2
      complex(ki) :: Qspe2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2e1
      Qspl4 = dotproduct(Q,l4)
      Qspl5 = dotproduct(Q,l5)
      QspQ = dotproduct(Q,Q)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspe1 = dotproduct(Q,e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspk1 = dotproduct(Q,k1)
      Qspk2 = dotproduct(Q,k2)
      Qspe2 = dotproduct(Q,e2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      acc258(1)=abb258(6)
      acc258(2)=abb258(7)
      acc258(3)=abb258(8)
      acc258(4)=abb258(9)
      acc258(5)=abb258(10)
      acc258(6)=abb258(11)
      acc258(7)=abb258(12)
      acc258(8)=abb258(13)
      acc258(9)=abb258(14)
      acc258(10)=abb258(15)
      acc258(11)=abb258(16)
      acc258(12)=abb258(17)
      acc258(13)=abb258(18)
      acc258(14)=abb258(19)
      acc258(15)=abb258(20)
      acc258(16)=abb258(21)
      acc258(17)=abb258(22)
      acc258(18)=abb258(23)
      acc258(19)=abb258(24)
      acc258(20)=abb258(25)
      acc258(21)=abb258(26)
      acc258(22)=abb258(27)
      acc258(23)=abb258(28)
      acc258(24)=abb258(29)
      acc258(25)=abb258(30)
      acc258(26)=abb258(31)
      acc258(27)=abb258(32)
      acc258(28)=abb258(33)
      acc258(29)=abb258(34)
      acc258(30)=abb258(35)
      acc258(31)=abb258(36)
      acc258(32)=abb258(37)
      acc258(33)=abb258(38)
      acc258(34)=abb258(39)
      acc258(35)=abb258(40)
      acc258(36)=abb258(41)
      acc258(37)=abb258(42)
      acc258(38)=abb258(43)
      acc258(39)=abb258(44)
      acc258(40)=abb258(45)
      acc258(41)=abb258(46)
      acc258(42)=abb258(47)
      acc258(43)=abb258(48)
      acc258(44)=abb258(49)
      acc258(45)=abb258(50)
      acc258(46)=abb258(51)
      acc258(47)=abb258(52)
      acc258(48)=abb258(53)
      acc258(49)=abb258(54)
      acc258(50)=abb258(55)
      acc258(51)=abb258(56)
      acc258(52)=abb258(57)
      acc258(53)=abb258(58)
      acc258(54)=abb258(59)
      acc258(55)=abb258(60)
      acc258(56)=abb258(61)
      acc258(57)=abb258(62)
      acc258(58)=abb258(63)
      acc258(59)=abb258(64)
      acc258(60)=abb258(65)
      acc258(61)=abb258(66)
      acc258(62)=abb258(67)
      acc258(63)=abb258(68)
      acc258(64)=abb258(69)
      acc258(65)=abb258(70)
      acc258(66)=abb258(71)
      acc258(67)=abb258(72)
      acc258(68)=abb258(73)
      acc258(69)=abb258(74)
      acc258(70)=abb258(75)
      acc258(71)=abb258(76)
      acc258(72)=abb258(77)
      acc258(73)=abb258(78)
      acc258(74)=abb258(79)
      acc258(75)=abb258(80)
      acc258(76)=abb258(81)
      acc258(77)=abb258(82)
      acc258(78)=abb258(83)
      acc258(79)=abb258(84)
      acc258(80)=abb258(85)
      acc258(81)=abb258(86)
      acc258(82)=abb258(87)
      acc258(83)=abb258(89)
      acc258(84)=abb258(90)
      acc258(85)=abb258(91)
      acc258(86)=abb258(92)
      acc258(87)=abb258(93)
      acc258(88)=abb258(94)
      acc258(89)=abb258(95)
      acc258(90)=abb258(96)
      acc258(91)=abb258(98)
      acc258(92)=abb258(100)
      acc258(93)=abb258(101)
      acc258(94)=abb258(103)
      acc258(95)=abb258(106)
      acc258(96)=abb258(108)
      acc258(97)=abb258(110)
      acc258(98)=abb258(112)
      acc258(99)=abb258(114)
      acc258(100)=abb258(115)
      acc258(101)=abb258(116)
      acc258(102)=abb258(118)
      acc258(103)=abb258(119)
      acc258(104)=abb258(120)
      acc258(105)=abb258(122)
      acc258(106)=abb258(123)
      acc258(107)=abb258(124)
      acc258(108)=abb258(125)
      acc258(109)=abb258(126)
      acc258(110)=abb258(127)
      acc258(111)=abb258(131)
      acc258(112)=abb258(132)
      acc258(113)=abb258(133)
      acc258(114)=abb258(134)
      acc258(115)=abb258(136)
      acc258(116)=abb258(137)
      acc258(117)=abb258(138)
      acc258(118)=abb258(139)
      acc258(119)=abb258(140)
      acc258(120)=abb258(141)
      acc258(121)=abb258(142)
      acc258(122)=abb258(143)
      acc258(123)=abb258(144)
      acc258(124)=abb258(145)
      acc258(125)=abb258(146)
      acc258(126)=abb258(147)
      acc258(127)=abb258(148)
      acc258(128)=abb258(149)
      acc258(129)=abb258(150)
      acc258(130)=abb258(151)
      acc258(131)=abb258(152)
      acc258(132)=abb258(154)
      acc258(133)=abb258(155)
      acc258(134)=abb258(156)
      acc258(135)=abb258(157)
      acc258(136)=abb258(158)
      acc258(137)=abb258(159)
      acc258(138)=Qspl4-Qspl5
      acc258(139)=-QspQ+acc258(138)
      acc258(139)=acc258(105)*acc258(139)
      acc258(140)=Qspvak1l4*acc258(74)
      acc258(141)=Qspvak1l5*acc258(72)
      acc258(142)=Qspvak2l5*acc258(49)
      acc258(143)=Qspvak2l4*acc258(33)
      acc258(139)=acc258(143)+acc258(142)+acc258(141)+acc258(66)+acc258(140)+ac&
      &c258(139)
      acc258(139)=Qspe1*acc258(139)
      acc258(140)=Qspvak2e1*acc258(109)
      acc258(141)=-Qspvae1l5*acc258(119)
      acc258(142)=Qspvae1l4*acc258(125)
      acc258(140)=acc258(142)+acc258(141)+acc258(25)+acc258(140)
      acc258(140)=QspQ*acc258(140)
      acc258(141)=acc258(63)*Qspval5e1
      acc258(142)=Qspvae1k2*acc258(34)
      acc258(141)=acc258(142)+acc258(83)+acc258(141)
      acc258(141)=Qspvak2l5*acc258(141)
      acc258(142)=Qspval4e1*acc258(109)
      acc258(143)=Qspvae1k2*acc258(80)
      acc258(142)=acc258(143)+acc258(6)+acc258(142)
      acc258(142)=Qspvak2l4*acc258(142)
      acc258(143)=acc258(59)*Qspval5k1
      acc258(144)=Qspvak1e1*acc258(67)
      acc258(145)=Qspvae1k1*acc258(94)
      acc258(146)=Qspval5e1*acc258(91)
      acc258(147)=Qspvak1k2*acc258(73)
      acc258(148)=Qspval4k1*acc258(64)
      acc258(149)=Qspval4l5*acc258(137)
      acc258(150)=Qspvak2e1*acc258(68)
      acc258(151)=Qspvae1l5*acc258(89)
      acc258(152)=Qspval5l4*acc258(136)
      acc258(153)=Qspval4e1*acc258(127)
      acc258(154)=Qspvak1l4*acc258(19)
      acc258(155)=Qspvae1k2*acc258(92)
      acc258(156)=Qspvae1l4*acc258(123)
      acc258(157)=Qspvae1l5*acc258(121)
      acc258(157)=acc258(101)+acc258(157)
      acc258(157)=Qspl5*acc258(157)
      acc258(158)=acc258(119)*Qspvae1k1
      acc258(158)=acc258(50)+acc258(158)
      acc258(158)=Qspvak1l5*acc258(158)
      acc258(159)=-acc258(63)*Qspvak1e1
      acc258(159)=acc258(42)+acc258(159)
      acc258(159)=Qspvak2k1*acc258(159)
      acc258(160)=-Qspvae1l4*acc258(12)
      acc258(160)=acc258(76)+acc258(160)
      acc258(160)=Qspl4*acc258(160)
      acc258(161)=Qspk1*acc258(40)
      acc258(162)=Qspvak2e1*acc258(52)
      acc258(162)=acc258(55)+acc258(162)
      acc258(162)=Qspk2*acc258(162)
      acc258(139)=acc258(139)+acc258(140)+acc258(162)+acc258(142)+acc258(141)+a&
      &cc258(161)+acc258(160)+acc258(159)+acc258(158)+acc258(157)+acc258(156)+a&
      &cc258(155)+acc258(154)+acc258(153)+acc258(152)+acc258(151)+acc258(150)+a&
      &cc258(149)+acc258(148)+acc258(147)+acc258(146)+acc258(145)+acc258(144)+a&
      &cc258(143)+acc258(32)
      acc258(139)=Qspe2*acc258(139)
      acc258(140)=Qspvae2l4*acc258(4)
      acc258(141)=-acc258(107)*Qspvak2e2
      acc258(142)=-Qspvae2l5*acc258(100)
      acc258(140)=acc258(142)+acc258(141)+acc258(11)+acc258(140)
      acc258(140)=QspQ*acc258(140)
      acc258(141)=Qspval5e2*acc258(107)
      acc258(142)=Qspvae2k2*acc258(116)
      acc258(141)=acc258(142)+acc258(113)+acc258(141)
      acc258(141)=Qspvak2l5*acc258(141)
      acc258(142)=acc258(57)*Qspval4e2
      acc258(143)=-Qspvae2k2*acc258(87)
      acc258(142)=acc258(143)+acc258(142)+acc258(14)
      acc258(142)=Qspvak2l4*acc258(142)
      acc258(143)=Qspvak1k2*acc258(77)
      acc258(144)=Qspval4k1*acc258(23)
      acc258(145)=Qspval4l5*acc258(10)
      acc258(146)=Qspvak2e2*acc258(47)
      acc258(147)=Qspvae2l4*acc258(111)
      acc258(148)=Qspval5l4*acc258(104)
      acc258(149)=Qspvak1e2*acc258(133)
      acc258(150)=Qspvae2k1*acc258(99)
      acc258(151)=Qspval5e2*acc258(117)
      acc258(152)=Qspvak1l4*acc258(41)
      acc258(153)=Qspvae2k2*acc258(112)
      acc258(154)=Qspvae2l5*acc258(95)
      acc258(155)=-Qspvae2l5*acc258(98)
      acc258(155)=acc258(78)+acc258(155)
      acc258(155)=Qspl5*acc258(155)
      acc258(156)=Qspvae2k1*acc258(100)
      acc258(156)=acc258(69)+acc258(156)
      acc258(156)=Qspvak1l5*acc258(156)
      acc258(157)=-Qspvak1e2*acc258(107)
      acc258(157)=acc258(24)+acc258(157)
      acc258(157)=Qspvak2k1*acc258(157)
      acc258(158)=Qspvae2l4*acc258(120)
      acc258(158)=acc258(58)+acc258(158)
      acc258(158)=Qspl4*acc258(158)
      acc258(159)=Qspvak2e2*acc258(2)
      acc258(159)=acc258(39)+acc258(159)
      acc258(159)=Qspk2*acc258(159)
      acc258(140)=acc258(140)+acc258(159)+acc258(142)+acc258(141)+acc258(158)+a&
      &cc258(157)+acc258(156)+acc258(155)+acc258(154)+acc258(153)+acc258(152)+a&
      &cc258(151)+acc258(150)+acc258(149)+acc258(148)+acc258(147)+acc258(146)+a&
      &cc258(145)+acc258(144)+acc258(46)+acc258(143)
      acc258(140)=Qspe1*acc258(140)
      acc258(138)=acc258(84)*acc258(138)
      acc258(141)=Qspk2-Qspk1
      acc258(141)=acc258(21)*acc258(141)
      acc258(142)=-Qspvae1e2*acc258(17)
      acc258(143)=-Qspvae2e1*acc258(20)
      acc258(144)=Qspvak1e2*acc258(134)
      acc258(145)=Qspvae2k1*acc258(108)
      acc258(146)=Qspval4e1*acc258(128)
      acc258(147)=Qspval5e2*acc258(118)
      acc258(148)=-Qspvak1l4*acc258(82)
      acc258(149)=Qspvae1k2*acc258(96)
      acc258(150)=Qspvae2k2*acc258(131)
      acc258(151)=Qspvae1l4*acc258(124)
      acc258(152)=Qspvae2l5*acc258(106)
      acc258(153)=-Qspvak1l5*acc258(44)
      acc258(154)=Qspvak2k1*acc258(36)
      acc258(155)=Qspvak2l5*acc258(85)
      acc258(156)=Qspvak2l4*acc258(71)
      acc258(157)=QspQ*acc258(13)
      acc258(138)=acc258(157)+acc258(156)+acc258(155)+acc258(154)+acc258(153)+a&
      &cc258(152)+acc258(151)+acc258(150)+acc258(149)+acc258(148)+acc258(147)+a&
      &cc258(146)+acc258(145)+acc258(144)+acc258(143)+acc258(8)+acc258(142)+acc&
      &258(141)+acc258(138)
      acc258(138)=QspQ*acc258(138)
      acc258(141)=Qspvae1e2*acc258(75)
      acc258(142)=Qspvae2e1*acc258(43)
      acc258(143)=Qspvak1e2*acc258(26)
      acc258(144)=Qspvae2k1*acc258(27)
      acc258(145)=Qspval4e1*acc258(7)
      acc258(146)=Qspval5e2*acc258(16)
      acc258(147)=Qspvae1k2*acc258(3)
      acc258(148)=Qspvae2k2*acc258(22)
      acc258(149)=Qspvae1l4*acc258(1)
      acc258(150)=Qspvae2l5*acc258(81)
      acc258(141)=acc258(141)+acc258(142)-acc258(147)-acc258(148)-acc258(149)-a&
      &cc258(150)-acc258(143)+acc258(144)+acc258(145)+acc258(146)
      acc258(142)=-acc258(30)+acc258(141)
      acc258(142)=Qspk1*acc258(142)
      acc258(143)=Qspvak2k1*acc258(29)
      acc258(144)=Qspvak2l5*acc258(54)
      acc258(145)=Qspvak2l4*acc258(51)
      acc258(141)=acc258(145)+acc258(144)+acc258(143)+acc258(115)-acc258(141)
      acc258(141)=Qspk2*acc258(141)
      acc258(143)=Qspvak1k2*acc258(37)
      acc258(144)=Qspval4k1*acc258(88)
      acc258(145)=Qspval4l5*acc258(9)
      acc258(146)=Qspl5*acc258(62)
      acc258(147)=Qspvak1l5*acc258(56)
      acc258(148)=Qspvak2k1*acc258(5)
      acc258(149)=Qspl4*acc258(70)
      acc258(150)=Qspvak2l5*acc258(90)
      acc258(143)=acc258(150)+acc258(149)+acc258(148)+acc258(147)+acc258(146)+a&
      &cc258(145)+acc258(144)+acc258(53)+acc258(143)
      acc258(143)=Qspvak2l4*acc258(143)
      acc258(144)=Qspvak1l4*acc258(86)
      acc258(145)=Qspl5*acc258(110)
      acc258(146)=Qspvak1l5*acc258(38)
      acc258(144)=acc258(146)+acc258(145)+acc258(48)+acc258(144)
      acc258(144)=Qspl4*acc258(144)
      acc258(145)=-Qspval5l4*acc258(130)
      acc258(146)=Qspl5*acc258(103)
      acc258(147)=Qspl4*acc258(102)
      acc258(145)=acc258(147)+acc258(146)+acc258(35)+acc258(145)
      acc258(145)=Qspvak2l5*acc258(145)
      acc258(146)=Qspvak1l4*acc258(130)
      acc258(147)=Qspvak1l5*acc258(79)
      acc258(146)=acc258(147)+acc258(28)+acc258(146)
      acc258(146)=Qspvak2k1*acc258(146)
      acc258(147)=Qspval5l4*acc258(135)
      acc258(148)=Qspvae1e2*acc258(65)
      acc258(149)=Qspvae2e1*acc258(15)
      acc258(150)=Qspvak1e2*acc258(132)
      acc258(151)=Qspvae2k1*acc258(61)
      acc258(152)=Qspval4e1*acc258(126)
      acc258(153)=Qspval5e2*acc258(114)
      acc258(154)=Qspvak1l4*acc258(18)
      acc258(155)=Qspvae1k2*acc258(60)
      acc258(156)=Qspvae2k2*acc258(129)
      acc258(157)=Qspvae1l4*acc258(122)
      acc258(158)=Qspvae2l5*acc258(93)
      acc258(159)=Qspl5*acc258(97)
      acc258(160)=Qspvak1l5*acc258(45)
      brack=acc258(31)+acc258(138)+acc258(139)+acc258(140)+acc258(141)+acc258(1&
      &42)+acc258(143)+acc258(144)+acc258(145)+acc258(146)+acc258(147)+acc258(1&
      &48)+acc258(149)+acc258(150)+acc258(151)+acc258(152)+acc258(153)+acc258(1&
      &54)+acc258(155)+acc258(156)+acc258(157)+acc258(158)+acc258(159)+acc258(1&
      &60)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d258h12l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd258h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d258
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d258 = 0.0_ki
      d258 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d258, ki), aimag(d258), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d258h12l1_qp
