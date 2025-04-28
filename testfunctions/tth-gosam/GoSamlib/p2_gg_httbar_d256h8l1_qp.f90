module     p2_gg_httbar_d256h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d256h8l1_qp.f90
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
      use p2_gg_httbar_abbrevd256h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc256(175)
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspl4
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspl5
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      complex(ki) :: Qspe1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspk1
      complex(ki) :: Qspe2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval5e2
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspl4 = dotproduct(Q,l4)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspl5 = dotproduct(Q,l5)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      Qspe1 = dotproduct(Q,e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspk1 = dotproduct(Q,k1)
      Qspe2 = dotproduct(Q,e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      acc256(1)=abb256(6)
      acc256(2)=abb256(7)
      acc256(3)=abb256(8)
      acc256(4)=abb256(9)
      acc256(5)=abb256(10)
      acc256(6)=abb256(11)
      acc256(7)=abb256(12)
      acc256(8)=abb256(13)
      acc256(9)=abb256(14)
      acc256(10)=abb256(15)
      acc256(11)=abb256(16)
      acc256(12)=abb256(17)
      acc256(13)=abb256(18)
      acc256(14)=abb256(19)
      acc256(15)=abb256(20)
      acc256(16)=abb256(21)
      acc256(17)=abb256(22)
      acc256(18)=abb256(23)
      acc256(19)=abb256(24)
      acc256(20)=abb256(25)
      acc256(21)=abb256(26)
      acc256(22)=abb256(27)
      acc256(23)=abb256(28)
      acc256(24)=abb256(29)
      acc256(25)=abb256(30)
      acc256(26)=abb256(31)
      acc256(27)=abb256(32)
      acc256(28)=abb256(33)
      acc256(29)=abb256(34)
      acc256(30)=abb256(35)
      acc256(31)=abb256(36)
      acc256(32)=abb256(37)
      acc256(33)=abb256(38)
      acc256(34)=abb256(39)
      acc256(35)=abb256(40)
      acc256(36)=abb256(41)
      acc256(37)=abb256(42)
      acc256(38)=abb256(43)
      acc256(39)=abb256(44)
      acc256(40)=abb256(45)
      acc256(41)=abb256(46)
      acc256(42)=abb256(47)
      acc256(43)=abb256(48)
      acc256(44)=abb256(49)
      acc256(45)=abb256(50)
      acc256(46)=abb256(51)
      acc256(47)=abb256(52)
      acc256(48)=abb256(53)
      acc256(49)=abb256(54)
      acc256(50)=abb256(55)
      acc256(51)=abb256(56)
      acc256(52)=abb256(57)
      acc256(53)=abb256(58)
      acc256(54)=abb256(59)
      acc256(55)=abb256(60)
      acc256(56)=abb256(62)
      acc256(57)=abb256(63)
      acc256(58)=abb256(64)
      acc256(59)=abb256(65)
      acc256(60)=abb256(66)
      acc256(61)=abb256(67)
      acc256(62)=abb256(68)
      acc256(63)=abb256(69)
      acc256(64)=abb256(70)
      acc256(65)=abb256(71)
      acc256(66)=abb256(72)
      acc256(67)=abb256(73)
      acc256(68)=abb256(74)
      acc256(69)=abb256(76)
      acc256(70)=abb256(77)
      acc256(71)=abb256(78)
      acc256(72)=abb256(79)
      acc256(73)=abb256(80)
      acc256(74)=abb256(81)
      acc256(75)=abb256(82)
      acc256(76)=abb256(83)
      acc256(77)=abb256(84)
      acc256(78)=abb256(85)
      acc256(79)=abb256(86)
      acc256(80)=abb256(87)
      acc256(81)=abb256(88)
      acc256(82)=abb256(90)
      acc256(83)=abb256(91)
      acc256(84)=abb256(92)
      acc256(85)=abb256(93)
      acc256(86)=abb256(94)
      acc256(87)=abb256(95)
      acc256(88)=abb256(96)
      acc256(89)=abb256(97)
      acc256(90)=abb256(98)
      acc256(91)=abb256(99)
      acc256(92)=abb256(100)
      acc256(93)=abb256(101)
      acc256(94)=abb256(102)
      acc256(95)=abb256(103)
      acc256(96)=abb256(104)
      acc256(97)=abb256(105)
      acc256(98)=abb256(106)
      acc256(99)=abb256(107)
      acc256(100)=abb256(108)
      acc256(101)=abb256(109)
      acc256(102)=abb256(110)
      acc256(103)=abb256(112)
      acc256(104)=abb256(113)
      acc256(105)=abb256(114)
      acc256(106)=abb256(115)
      acc256(107)=abb256(116)
      acc256(108)=abb256(117)
      acc256(109)=abb256(118)
      acc256(110)=abb256(119)
      acc256(111)=abb256(121)
      acc256(112)=abb256(122)
      acc256(113)=abb256(124)
      acc256(114)=abb256(127)
      acc256(115)=abb256(128)
      acc256(116)=abb256(130)
      acc256(117)=abb256(131)
      acc256(118)=abb256(132)
      acc256(119)=abb256(133)
      acc256(120)=abb256(135)
      acc256(121)=abb256(136)
      acc256(122)=abb256(137)
      acc256(123)=abb256(138)
      acc256(124)=abb256(139)
      acc256(125)=abb256(140)
      acc256(126)=abb256(141)
      acc256(127)=abb256(142)
      acc256(128)=abb256(147)
      acc256(129)=abb256(148)
      acc256(130)=abb256(149)
      acc256(131)=abb256(150)
      acc256(132)=abb256(151)
      acc256(133)=abb256(152)
      acc256(134)=abb256(153)
      acc256(135)=abb256(154)
      acc256(136)=abb256(155)
      acc256(137)=abb256(156)
      acc256(138)=abb256(157)
      acc256(139)=abb256(158)
      acc256(140)=abb256(159)
      acc256(141)=abb256(161)
      acc256(142)=abb256(162)
      acc256(143)=abb256(163)
      acc256(144)=abb256(164)
      acc256(145)=abb256(165)
      acc256(146)=abb256(166)
      acc256(147)=abb256(167)
      acc256(148)=abb256(168)
      acc256(149)=abb256(170)
      acc256(150)=abb256(171)
      acc256(151)=Qspvak1l5*acc256(98)
      acc256(152)=Qspvak2k1*acc256(74)
      acc256(153)=Qspval4l5*acc256(33)
      acc256(154)=Qspl4*acc256(125)
      acc256(155)=Qspvak1k2*acc256(40)
      acc256(156)=Qspval4k1*acc256(102)
      acc256(157)=Qspl5*acc256(122)
      acc256(158)=Qspval4k2*acc256(36)
      acc256(159)=Qspvak2l5*acc256(72)
      acc256(160)=Qspk2*acc256(85)
      acc256(161)=QspQ*acc256(59)
      acc256(151)=acc256(161)+acc256(160)+acc256(159)+acc256(158)+acc256(157)+a&
      &cc256(156)+acc256(155)+acc256(154)+acc256(153)+acc256(152)+acc256(1)+acc&
      &256(151)
      acc256(151)=Qspe1*acc256(151)
      acc256(152)=-Qspval4e1*acc256(99)
      acc256(153)=Qspvak2e1*acc256(143)
      acc256(154)=Qspvae1l5*acc256(119)
      acc256(155)=-Qspvae1k2*acc256(91)
      acc256(152)=acc256(155)+acc256(154)+acc256(153)+acc256(35)+acc256(152)
      acc256(152)=QspQ*acc256(152)
      acc256(153)=-acc256(91)*Qspvae1l4
      acc256(154)=-Qspvak2e1*acc256(145)
      acc256(153)=acc256(154)+acc256(49)+acc256(153)
      acc256(153)=Qspval4k2*acc256(153)
      acc256(154)=-Qspval5e1*acc256(143)
      acc256(155)=Qspvae1k2*acc256(8)
      acc256(154)=acc256(155)+acc256(14)+acc256(154)
      acc256(154)=Qspvak2l5*acc256(154)
      acc256(155)=-Qspvak2e1*acc256(93)
      acc256(156)=Qspvae1k2*acc256(57)
      acc256(155)=acc256(156)+acc256(48)+acc256(155)
      acc256(155)=Qspk2*acc256(155)
      acc256(156)=acc256(34)*Qspvak1l4
      acc256(157)=Qspvak1e1*acc256(101)
      acc256(158)=Qspvae1k1*acc256(147)
      acc256(159)=-Qspvae1l4*acc256(80)
      acc256(160)=Qspval5k1*acc256(115)
      acc256(161)=-Qspval5k2*acc256(82)
      acc256(162)=Qspval4e1*acc256(100)
      acc256(163)=Qspvak2l4*acc256(81)
      acc256(164)=Qspvak2e1*acc256(23)
      acc256(165)=Qspval5e1*acc256(76)
      acc256(166)=Qspvak1l5*acc256(77)
      acc256(167)=Qspvak2k1*acc256(51)
      acc256(168)=Qspval4l5*acc256(97)
      acc256(169)=Qspvae1l5*acc256(88)
      acc256(170)=Qspvae1k2*acc256(58)
      acc256(171)=Qspval4e1*acc256(138)
      acc256(171)=acc256(104)+acc256(171)
      acc256(171)=Qspl4*acc256(171)
      acc256(172)=acc256(91)*Qspvae1k1
      acc256(172)=acc256(78)+acc256(172)
      acc256(172)=Qspvak1k2*acc256(172)
      acc256(173)=acc256(99)*Qspvak1e1
      acc256(173)=acc256(41)+acc256(173)
      acc256(173)=Qspval4k1*acc256(173)
      acc256(174)=-Qspvae1l5*acc256(60)
      acc256(174)=acc256(114)+acc256(174)
      acc256(174)=Qspl5*acc256(174)
      acc256(175)=Qspk1*acc256(50)
      acc256(151)=acc256(151)+acc256(152)+acc256(155)+acc256(154)+acc256(175)+a&
      &cc256(153)+acc256(174)+acc256(173)+acc256(172)+acc256(171)+acc256(170)+a&
      &cc256(169)+acc256(168)+acc256(167)+acc256(166)+acc256(165)+acc256(164)+a&
      &cc256(163)+acc256(162)+acc256(161)+acc256(160)+acc256(159)+acc256(158)+a&
      &cc256(157)+acc256(69)+acc256(156)
      acc256(151)=Qspe2*acc256(151)
      acc256(152)=Qspval4l5*acc256(131)
      acc256(153)=-Qspvae1e2*acc256(44)
      acc256(154)=Qspvae2e1*acc256(24)
      acc256(155)=Qspvak1e2*acc256(146)
      acc256(156)=Qspvae2k1*acc256(46)
      acc256(157)=Qspvae2l4*acc256(134)
      acc256(158)=Qspval5e1*acc256(132)
      acc256(159)=Qspvak1l5*acc256(13)
      acc256(160)=-Qspvak2k1*acc256(130)
      acc256(161)=Qspval4e2*acc256(137)
      acc256(162)=Qspvae1l5*acc256(9)
      acc256(163)=Qspvae1k2*acc256(75)
      acc256(164)=Qspvak2e2*acc256(7)
      acc256(165)=-Qspl4*acc256(32)
      acc256(166)=Qspvak1k2*acc256(65)
      acc256(167)=Qspval4k1*acc256(109)
      acc256(168)=-Qspl5*acc256(127)
      acc256(169)=Qspval4k2*acc256(150)
      acc256(170)=-Qspk1*acc256(22)
      acc256(171)=Qspvak2l5*acc256(73)
      acc256(172)=Qspk2*acc256(25)
      acc256(173)=QspQ*acc256(27)
      acc256(153)=acc256(173)+acc256(172)+acc256(171)+acc256(170)+acc256(169)+a&
      &cc256(168)+acc256(167)+acc256(166)+acc256(165)+acc256(164)+acc256(163)+a&
      &cc256(162)+acc256(161)+acc256(152)+acc256(160)+acc256(159)+acc256(158)+a&
      &cc256(157)+acc256(156)+acc256(155)+acc256(154)+acc256(12)+acc256(153)
      acc256(153)=QspQ*acc256(153)
      acc256(154)=Qspvae2l5*acc256(103)
      acc256(155)=-Qspvae2k2*acc256(113)
      acc256(156)=-Qspval4e2*acc256(117)
      acc256(157)=Qspvak2e2*acc256(47)
      acc256(154)=acc256(157)+acc256(156)+acc256(155)+acc256(39)+acc256(154)
      acc256(154)=QspQ*acc256(154)
      acc256(155)=-Qspvae2l4*acc256(113)
      acc256(156)=-Qspvak2e2*acc256(116)
      acc256(155)=acc256(156)+acc256(38)+acc256(155)
      acc256(155)=Qspval4k2*acc256(155)
      acc256(156)=-acc256(47)*Qspval5e2
      acc256(157)=Qspvae2k2*acc256(2)
      acc256(156)=acc256(157)+acc256(19)+acc256(156)
      acc256(156)=Qspvak2l5*acc256(156)
      acc256(157)=Qspvae2k2*acc256(140)
      acc256(158)=Qspvak2e2*acc256(128)
      acc256(157)=acc256(158)+acc256(129)+acc256(157)
      acc256(157)=Qspk2*acc256(157)
      acc256(158)=Qspval5k1*acc256(68)
      acc256(159)=Qspval5k2*acc256(107)
      acc256(160)=Qspvae2l5*acc256(94)
      acc256(161)=Qspvak2l4*acc256(92)
      acc256(162)=Qspvae2k2*acc256(139)
      acc256(163)=Qspvak1e2*acc256(108)
      acc256(164)=Qspvae2k1*acc256(144)
      acc256(165)=Qspvae2l4*acc256(89)
      acc256(166)=Qspvak1l5*acc256(96)
      acc256(167)=Qspvak2k1*acc256(29)
      acc256(168)=Qspval4l5*acc256(148)
      acc256(169)=Qspval4e2*acc256(112)
      acc256(170)=Qspvak2e2*acc256(142)
      acc256(171)=-Qspval4e2*acc256(16)
      acc256(171)=acc256(110)+acc256(171)
      acc256(171)=Qspl4*acc256(171)
      acc256(172)=Qspvae2k1*acc256(113)
      acc256(172)=acc256(37)+acc256(172)
      acc256(172)=Qspvak1k2*acc256(172)
      acc256(173)=Qspvak1e2*acc256(117)
      acc256(173)=acc256(70)+acc256(173)
      acc256(173)=Qspval4k1*acc256(173)
      acc256(174)=Qspvae2l5*acc256(105)
      acc256(174)=acc256(120)+acc256(174)
      acc256(174)=Qspl5*acc256(174)
      acc256(154)=acc256(154)+acc256(157)+acc256(156)+acc256(155)+acc256(174)+a&
      &cc256(173)+acc256(172)+acc256(171)+acc256(170)+acc256(169)+acc256(168)+a&
      &cc256(167)+acc256(166)+acc256(165)+acc256(164)+acc256(163)+acc256(162)+a&
      &cc256(161)+acc256(160)+acc256(159)+acc256(61)+acc256(158)
      acc256(154)=Qspe1*acc256(154)
      acc256(155)=Qspvae1e2*acc256(84)
      acc256(156)=Qspvae2e1*acc256(71)
      acc256(157)=Qspvak1e2*acc256(17)
      acc256(158)=Qspvae2k1*acc256(20)
      acc256(159)=Qspvae2l4*acc256(54)
      acc256(160)=Qspval5e1*acc256(18)
      acc256(161)=Qspval4e2*acc256(135)
      acc256(162)=Qspvae1l5*acc256(111)
      acc256(163)=Qspvae1k2*acc256(11)
      acc256(164)=Qspvak2e2*acc256(10)
      acc256(155)=-acc256(160)-acc256(161)-acc256(162)+acc256(163)-acc256(156)+&
      &acc256(157)+acc256(158)-acc256(159)+acc256(155)-acc256(164)
      acc256(156)=acc256(5)+acc256(155)
      acc256(156)=Qspk1*acc256(156)
      acc256(157)=Qspvak2k1*acc256(87)
      acc256(158)=Qspl4*acc256(126)
      acc256(159)=Qspval4k1*acc256(6)
      acc256(160)=-Qspl5*acc256(121)
      acc256(161)=Qspk2*acc256(3)
      acc256(155)=acc256(161)+acc256(160)+acc256(159)+acc256(158)+acc256(157)+a&
      &cc256(30)-acc256(155)
      acc256(155)=Qspk2*acc256(155)
      acc256(157)=Qspval5k1*acc256(130)
      acc256(158)=Qspval5k2*acc256(86)
      acc256(159)=Qspl4*acc256(42)
      acc256(160)=Qspvak1k2*acc256(21)
      acc256(161)=Qspval4k1*acc256(90)
      acc256(162)=Qspval4k2*acc256(31)
      acc256(157)=acc256(162)+acc256(161)+acc256(160)+acc256(159)+acc256(158)+a&
      &cc256(63)+acc256(157)
      acc256(157)=Qspvak2l5*acc256(157)
      acc256(158)=Qspvak1l5*acc256(106)
      acc256(159)=Qspvak1k2*acc256(43)
      acc256(158)=acc256(159)-acc256(152)+acc256(28)+acc256(158)
      acc256(158)=Qspl5*acc256(158)
      acc256(159)=-Qspvak2l4*acc256(52)
      acc256(160)=Qspl5*acc256(83)
      acc256(159)=acc256(160)+acc256(149)+acc256(159)
      acc256(159)=Qspval4k2*acc256(159)
      acc256(160)=Qspvak2l4*acc256(62)
      acc256(161)=Qspvae1e2*acc256(79)
      acc256(162)=Qspvae2e1*acc256(67)
      acc256(163)=Qspvak1e2*acc256(56)
      acc256(164)=Qspvae2k1*acc256(95)
      acc256(165)=Qspvae2l4*acc256(133)
      acc256(166)=Qspval5e1*acc256(123)
      acc256(167)=Qspvak1l5*acc256(26)
      acc256(168)=Qspvak2k1*acc256(45)
      acc256(169)=Qspval4l5*acc256(66)
      acc256(170)=Qspval4e2*acc256(136)
      acc256(171)=Qspvae1l5*acc256(118)
      acc256(172)=Qspvae1k2*acc256(15)
      acc256(173)=Qspvak2e2*acc256(141)
      acc256(152)=acc256(124)+acc256(152)
      acc256(152)=Qspl4*acc256(152)
      acc256(174)=Qspvak2k1*acc256(52)
      acc256(174)=acc256(53)+acc256(174)
      acc256(174)=Qspvak1k2*acc256(174)
      acc256(175)=Qspvak1l5*acc256(4)
      acc256(175)=acc256(64)+acc256(175)
      acc256(175)=Qspval4k1*acc256(175)
      brack=acc256(55)+acc256(151)+acc256(152)+acc256(153)+acc256(154)+acc256(1&
      &55)+acc256(156)+acc256(157)+acc256(158)+acc256(159)+acc256(160)+acc256(1&
      &61)+acc256(162)+acc256(163)+acc256(164)+acc256(165)+acc256(166)+acc256(1&
      &67)+acc256(168)+acc256(169)+acc256(170)+acc256(171)+acc256(172)+acc256(1&
      &73)+acc256(174)+acc256(175)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d256h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd256h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d256
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d256 = 0.0_ki
      d256 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d256, ki), aimag(d256), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d256h8l1_qp
