module     p2_gg_httbar_d264h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d264h0l1.f90
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
      use p2_gg_httbar_abbrevd264h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc264(187)
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspl5
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspk2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: QspQ
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspl4
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspe1
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspvae1k2
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspl5 = dotproduct(Q,l5)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspk2 = dotproduct(Q,k2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      QspQ = dotproduct(Q,Q)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspl4 = dotproduct(Q,l4)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspe1 = dotproduct(Q,e1)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      acc264(1)=abb264(8)
      acc264(2)=abb264(9)
      acc264(3)=abb264(10)
      acc264(4)=abb264(11)
      acc264(5)=abb264(12)
      acc264(6)=abb264(13)
      acc264(7)=abb264(14)
      acc264(8)=abb264(15)
      acc264(9)=abb264(16)
      acc264(10)=abb264(17)
      acc264(11)=abb264(18)
      acc264(12)=abb264(19)
      acc264(13)=abb264(20)
      acc264(14)=abb264(21)
      acc264(15)=abb264(22)
      acc264(16)=abb264(23)
      acc264(17)=abb264(24)
      acc264(18)=abb264(25)
      acc264(19)=abb264(26)
      acc264(20)=abb264(27)
      acc264(21)=abb264(28)
      acc264(22)=abb264(29)
      acc264(23)=abb264(30)
      acc264(24)=abb264(31)
      acc264(25)=abb264(32)
      acc264(26)=abb264(33)
      acc264(27)=abb264(34)
      acc264(28)=abb264(35)
      acc264(29)=abb264(36)
      acc264(30)=abb264(37)
      acc264(31)=abb264(38)
      acc264(32)=abb264(39)
      acc264(33)=abb264(40)
      acc264(34)=abb264(41)
      acc264(35)=abb264(42)
      acc264(36)=abb264(43)
      acc264(37)=abb264(44)
      acc264(38)=abb264(45)
      acc264(39)=abb264(46)
      acc264(40)=abb264(47)
      acc264(41)=abb264(48)
      acc264(42)=abb264(49)
      acc264(43)=abb264(50)
      acc264(44)=abb264(51)
      acc264(45)=abb264(52)
      acc264(46)=abb264(53)
      acc264(47)=abb264(54)
      acc264(48)=abb264(55)
      acc264(49)=abb264(56)
      acc264(50)=abb264(57)
      acc264(51)=abb264(58)
      acc264(52)=abb264(59)
      acc264(53)=abb264(60)
      acc264(54)=abb264(61)
      acc264(55)=abb264(62)
      acc264(56)=abb264(63)
      acc264(57)=abb264(64)
      acc264(58)=abb264(65)
      acc264(59)=abb264(66)
      acc264(60)=abb264(67)
      acc264(61)=abb264(68)
      acc264(62)=abb264(69)
      acc264(63)=abb264(70)
      acc264(64)=abb264(71)
      acc264(65)=abb264(72)
      acc264(66)=abb264(73)
      acc264(67)=abb264(74)
      acc264(68)=abb264(75)
      acc264(69)=abb264(76)
      acc264(70)=abb264(77)
      acc264(71)=abb264(78)
      acc264(72)=abb264(79)
      acc264(73)=abb264(80)
      acc264(74)=abb264(81)
      acc264(75)=abb264(82)
      acc264(76)=abb264(83)
      acc264(77)=abb264(84)
      acc264(78)=abb264(86)
      acc264(79)=abb264(87)
      acc264(80)=abb264(89)
      acc264(81)=abb264(90)
      acc264(82)=abb264(91)
      acc264(83)=abb264(92)
      acc264(84)=abb264(93)
      acc264(85)=abb264(94)
      acc264(86)=abb264(95)
      acc264(87)=abb264(96)
      acc264(88)=abb264(97)
      acc264(89)=abb264(98)
      acc264(90)=abb264(100)
      acc264(91)=abb264(102)
      acc264(92)=abb264(103)
      acc264(93)=abb264(104)
      acc264(94)=abb264(106)
      acc264(95)=abb264(107)
      acc264(96)=abb264(108)
      acc264(97)=abb264(109)
      acc264(98)=abb264(110)
      acc264(99)=abb264(111)
      acc264(100)=abb264(112)
      acc264(101)=abb264(114)
      acc264(102)=abb264(115)
      acc264(103)=abb264(116)
      acc264(104)=abb264(117)
      acc264(105)=abb264(118)
      acc264(106)=abb264(119)
      acc264(107)=abb264(121)
      acc264(108)=abb264(122)
      acc264(109)=abb264(123)
      acc264(110)=abb264(124)
      acc264(111)=abb264(125)
      acc264(112)=abb264(126)
      acc264(113)=abb264(127)
      acc264(114)=abb264(130)
      acc264(115)=abb264(131)
      acc264(116)=abb264(133)
      acc264(117)=abb264(134)
      acc264(118)=abb264(136)
      acc264(119)=abb264(138)
      acc264(120)=abb264(139)
      acc264(121)=abb264(141)
      acc264(122)=abb264(142)
      acc264(123)=abb264(143)
      acc264(124)=abb264(144)
      acc264(125)=abb264(147)
      acc264(126)=abb264(148)
      acc264(127)=abb264(149)
      acc264(128)=abb264(152)
      acc264(129)=abb264(157)
      acc264(130)=abb264(159)
      acc264(131)=abb264(160)
      acc264(132)=abb264(161)
      acc264(133)=abb264(163)
      acc264(134)=abb264(164)
      acc264(135)=abb264(165)
      acc264(136)=abb264(166)
      acc264(137)=abb264(168)
      acc264(138)=abb264(169)
      acc264(139)=abb264(171)
      acc264(140)=abb264(172)
      acc264(141)=abb264(173)
      acc264(142)=abb264(174)
      acc264(143)=abb264(175)
      acc264(144)=abb264(178)
      acc264(145)=abb264(179)
      acc264(146)=abb264(181)
      acc264(147)=abb264(183)
      acc264(148)=abb264(184)
      acc264(149)=abb264(192)
      acc264(150)=abb264(194)
      acc264(151)=abb264(196)
      acc264(152)=abb264(197)
      acc264(153)=abb264(198)
      acc264(154)=abb264(200)
      acc264(155)=abb264(203)
      acc264(156)=abb264(208)
      acc264(157)=Qspvak2e1*acc264(132)
      acc264(158)=Qspvak1e2*acc264(99)
      acc264(159)=Qspval3e1*acc264(37)
      acc264(160)=Qspvae1l3*acc264(152)
      acc264(161)=Qspval4e1*acc264(84)
      acc264(162)=Qspvae1l4*acc264(153)
      acc264(163)=-Qspl5*acc264(30)
      acc264(164)=Qspvak1k2*acc264(57)
      acc264(165)=Qspval5k1*acc264(107)
      acc264(166)=Qspvae2k1*acc264(71)
      acc264(167)=Qspk2*acc264(70)
      acc264(168)=Qspval5e2*acc264(114)
      acc264(169)=Qspvae2l4*acc264(148)
      acc264(170)=Qspvak2e2*acc264(137)
      acc264(171)=Qspval5k2*acc264(42)
      acc264(172)=Qspvae2k2*acc264(86)
      acc264(173)=Qspval4e2*acc264(44)
      acc264(174)=Qspvae2k2*acc264(127)
      acc264(174)=acc264(40)+acc264(174)
      acc264(174)=Qspvae1e2*acc264(174)
      acc264(175)=Qspval4e2*acc264(98)
      acc264(175)=acc264(67)+acc264(175)
      acc264(175)=Qspvae2e1*acc264(175)
      acc264(176)=QspQ*acc264(61)
      acc264(157)=acc264(176)+acc264(175)+acc264(174)+acc264(173)+acc264(172)+a&
      &cc264(171)+acc264(170)+acc264(169)+acc264(168)+acc264(167)+acc264(166)+a&
      &cc264(165)+acc264(164)+acc264(163)+acc264(162)+acc264(161)+acc264(160)+a&
      &cc264(159)+acc264(158)+acc264(24)+acc264(157)
      acc264(157)=QspQ*acc264(157)
      acc264(158)=acc264(131)*Qspval5l4
      acc264(159)=Qspl4*acc264(39)
      acc264(160)=Qspvae2k2*acc264(7)
      acc264(158)=acc264(160)+acc264(159)+acc264(32)+acc264(158)
      acc264(158)=Qspval4e2*acc264(158)
      acc264(159)=-Qspval5e2*acc264(131)
      acc264(160)=Qspval4e2*acc264(72)
      acc264(159)=acc264(160)+acc264(27)+acc264(159)
      acc264(159)=QspQ*acc264(159)
      acc264(160)=acc264(136)*Qspval4l5
      acc264(161)=acc264(74)*Qspval5l3
      acc264(162)=acc264(45)*Qspval3l5
      acc264(163)=acc264(38)*Qspvak2l5
      acc264(164)=Qspval5l4*acc264(47)
      acc264(165)=Qspvak2k1*acc264(19)
      acc264(166)=Qspval3k1*acc264(43)
      acc264(167)=Qspval3k2*acc264(35)
      acc264(168)=Qspval4k1*acc264(101)
      acc264(169)=Qspl4*acc264(54)
      acc264(170)=Qspval4k2*acc264(28)
      acc264(171)=Qspk2*acc264(65)
      acc264(172)=Qspval5e2*acc264(91)
      acc264(173)=Qspvae2l4*acc264(147)
      acc264(174)=Qspval4k2*acc264(80)
      acc264(174)=acc264(125)+acc264(174)
      acc264(174)=Qspvak2e2*acc264(174)
      acc264(175)=Qspvak2e2*acc264(52)
      acc264(175)=acc264(66)+acc264(175)
      acc264(175)=Qspval5k2*acc264(175)
      acc264(176)=Qspval5e2*acc264(102)
      acc264(176)=acc264(1)+acc264(176)
      acc264(176)=Qspvae2k2*acc264(176)
      acc264(158)=acc264(159)+acc264(158)+acc264(176)+acc264(175)+acc264(174)+a&
      &cc264(173)+acc264(172)+acc264(171)+acc264(170)+acc264(169)+acc264(168)+a&
      &cc264(167)+acc264(166)+acc264(165)+acc264(164)+acc264(5)+acc264(163)+acc&
      &264(162)+acc264(160)+acc264(161)
      acc264(158)=Qspe1*acc264(158)
      acc264(159)=-Qspvak1l3*acc264(142)
      acc264(160)=-Qspvak1l4*acc264(121)
      acc264(161)=Qspval4l3*acc264(8)
      acc264(162)=Qspl4*acc264(112)
      acc264(163)=Qspl5*acc264(111)
      acc264(164)=-Qspvak1k2*acc264(138)
      acc264(165)=Qspval5k1*acc264(113)
      acc264(166)=Qspvae2k1*acc264(100)
      acc264(167)=Qspval4k2*acc264(64)
      acc264(168)=Qspk2*acc264(81)
      acc264(169)=-Qspval4k2*acc264(127)
      acc264(169)=acc264(151)+acc264(169)
      acc264(169)=Qspvae2l4*acc264(169)
      acc264(170)=-Qspval5k2*acc264(60)
      acc264(171)=-Qspk2*acc264(51)
      acc264(171)=acc264(90)+acc264(171)
      acc264(171)=Qspvae2k2*acc264(171)
      acc264(159)=acc264(171)+acc264(170)+acc264(169)+acc264(168)+acc264(167)+a&
      &cc264(166)+acc264(165)+acc264(164)+acc264(163)+acc264(162)+acc264(161)+a&
      &cc264(160)+acc264(106)+acc264(159)
      acc264(159)=Qspvae1e2*acc264(159)
      acc264(160)=Qspvak2k1*acc264(9)
      acc264(161)=Qspval3k1*acc264(73)
      acc264(162)=Qspval3k2*acc264(6)
      acc264(163)=Qspval4k1*acc264(103)
      acc264(164)=Qspl4*acc264(89)
      acc264(165)=Qspvak1e2*acc264(120)
      acc264(166)=Qspvak1k2*acc264(63)
      acc264(167)=Qspval4k2*acc264(41)
      acc264(168)=Qspk2*acc264(12)
      acc264(169)=Qspval5e2*acc264(108)
      acc264(170)=Qspval4k2*acc264(85)
      acc264(170)=acc264(130)+acc264(170)
      acc264(170)=Qspvak2e2*acc264(170)
      acc264(171)=Qspval5k2*acc264(87)
      acc264(172)=Qspl4*acc264(13)
      acc264(172)=acc264(93)+acc264(172)
      acc264(172)=Qspval4e2*acc264(172)
      acc264(160)=acc264(172)+acc264(171)+acc264(170)+acc264(169)+acc264(168)+a&
      &cc264(167)+acc264(166)+acc264(165)+acc264(164)+acc264(163)+acc264(162)+a&
      &cc264(161)+acc264(48)+acc264(160)
      acc264(160)=Qspvae2e1*acc264(160)
      acc264(161)=Qspvae1k2*acc264(145)
      acc264(162)=Qspval3e1*acc264(46)
      acc264(163)=Qspvae1l3*acc264(128)
      acc264(164)=Qspval4e1*acc264(123)
      acc264(165)=Qspvae1l4*acc264(118)
      acc264(166)=Qspvae2l4*acc264(95)
      acc264(167)=Qspvak2e2*acc264(119)
      acc264(161)=acc264(167)+acc264(166)+acc264(165)+acc264(164)+acc264(163)+a&
      &cc264(162)+acc264(16)+acc264(161)
      acc264(161)=Qspval5k2*acc264(161)
      acc264(162)=Qspl4*acc264(33)
      acc264(163)=Qspl5*acc264(129)
      acc264(164)=Qspval5k1*acc264(116)
      acc264(165)=Qspvae2k1*acc264(139)
      acc264(166)=Qspval5k2*acc264(115)
      acc264(167)=Qspvae2k2*acc264(156)
      acc264(162)=acc264(167)+acc264(166)+acc264(165)+acc264(164)+acc264(163)+a&
      &cc264(3)+acc264(162)
      acc264(162)=Qspval4e2*acc264(162)
      acc264(163)=-Qspvak1e2*acc264(141)
      acc264(164)=Qspvak1k2*acc264(68)
      acc264(165)=Qspval4k2*acc264(31)
      acc264(166)=-Qspk2*acc264(79)
      acc264(167)=Qspval5e2*acc264(144)
      acc264(163)=acc264(167)+acc264(166)+acc264(165)+acc264(164)+acc264(117)+a&
      &cc264(163)
      acc264(163)=Qspvae2l4*acc264(163)
      acc264(164)=Qspvak2e1*acc264(77)
      acc264(165)=Qspval3e1*acc264(69)
      acc264(166)=Qspval4e1*acc264(104)
      acc264(167)=Qspk2*acc264(155)
      acc264(168)=Qspvak2e2*acc264(135)
      acc264(164)=acc264(168)+acc264(167)+acc264(166)+acc264(165)+acc264(56)+ac&
      &c264(164)
      acc264(164)=Qspvae2k2*acc264(164)
      acc264(165)=Qspl5*acc264(133)
      acc264(166)=Qspval5k1*acc264(124)
      acc264(167)=Qspvae2k1*acc264(105)
      acc264(168)=Qspval4k2*acc264(78)
      acc264(165)=acc264(168)+acc264(167)+acc264(166)+acc264(94)+acc264(165)
      acc264(165)=Qspvak2e2*acc264(165)
      acc264(166)=-Qspvak2e1*acc264(110)
      acc264(167)=-Qspval3e1*acc264(58)
      acc264(168)=Qspval4e1*acc264(83)
      acc264(166)=acc264(168)+acc264(167)+acc264(18)+acc264(166)
      acc264(166)=Qspl5*acc264(166)
      acc264(167)=Qspvae1k2*acc264(76)
      acc264(168)=Qspvae1l3*acc264(75)
      acc264(169)=Qspvae1l4*acc264(50)
      acc264(167)=acc264(169)+acc264(168)+acc264(34)+acc264(167)
      acc264(167)=Qspvak1k2*acc264(167)
      acc264(168)=Qspvak2e1*acc264(21)
      acc264(169)=Qspval3e1*acc264(82)
      acc264(170)=Qspval4e1*acc264(11)
      acc264(168)=acc264(170)+acc264(169)+acc264(49)+acc264(168)
      acc264(168)=Qspval5k1*acc264(168)
      acc264(169)=Qspvak2e1*acc264(88)
      acc264(170)=Qspval3e1*acc264(140)
      acc264(171)=Qspval4e1*acc264(53)
      acc264(169)=acc264(171)+acc264(170)+acc264(10)+acc264(169)
      acc264(169)=Qspvae2k1*acc264(169)
      acc264(170)=Qspvae1k2*acc264(2)
      acc264(171)=Qspvae1l3*acc264(149)
      acc264(172)=Qspvae1l4*acc264(154)
      acc264(170)=acc264(172)+acc264(171)+acc264(26)+acc264(170)
      acc264(170)=Qspval5e2*acc264(170)
      acc264(171)=Qspvae1l3*acc264(62)
      acc264(172)=Qspvae1l4*acc264(150)
      acc264(171)=acc264(172)+acc264(4)+acc264(171)
      acc264(171)=Qspk2*acc264(171)
      acc264(172)=Qspvak1l3*acc264(29)
      acc264(173)=Qspvak1l4*acc264(22)
      acc264(174)=Qspval4l3*acc264(134)
      acc264(175)=Qspvak2k1*acc264(14)
      acc264(176)=Qspval3k1*acc264(36)
      acc264(177)=Qspval3k2*acc264(15)
      acc264(178)=Qspval4k1*acc264(20)
      acc264(179)=Qspvae1k2*acc264(55)
      acc264(180)=Qspvak2e1*acc264(59)
      acc264(181)=Qspl4*acc264(23)
      acc264(182)=Qspvae1k2*acc264(146)
      acc264(182)=acc264(97)+acc264(182)
      acc264(182)=Qspvak1e2*acc264(182)
      acc264(183)=Qspval3e1*acc264(96)
      acc264(184)=Qspvak1e2*acc264(143)
      acc264(184)=acc264(126)+acc264(184)
      acc264(184)=Qspvae1l3*acc264(184)
      acc264(185)=Qspval4e1*acc264(92)
      acc264(186)=Qspvak1e2*acc264(122)
      acc264(186)=acc264(109)+acc264(186)
      acc264(186)=Qspvae1l4*acc264(186)
      acc264(187)=Qspval4k2*acc264(25)
      brack=acc264(17)+acc264(157)+acc264(158)+acc264(159)+acc264(160)+acc264(1&
      &61)+acc264(162)+acc264(163)+acc264(164)+acc264(165)+acc264(166)+acc264(1&
      &67)+acc264(168)+acc264(169)+acc264(170)+acc264(171)+acc264(172)+acc264(1&
      &73)+acc264(174)+acc264(175)+acc264(176)+acc264(177)+acc264(178)+acc264(1&
      &79)+acc264(180)+acc264(181)+acc264(182)+acc264(183)+acc264(184)+acc264(1&
      &85)+acc264(186)+acc264(187)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d264h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd264h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d264
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(+Q_ext(0:3),  ki_nin), aimag(+Q_ext(0:3)), ki)
      d264 = 0.0_ki
      d264 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d264, ki), aimag(d264), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d264h0l1
