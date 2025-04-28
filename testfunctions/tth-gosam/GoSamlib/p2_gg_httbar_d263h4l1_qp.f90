module     p2_gg_httbar_d263h4l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d263h4l1_qp.f90
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
      use p2_gg_httbar_abbrevd263h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc263(163)
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspl4
      complex(ki) :: Qspk2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: QspQ
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspe1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspl5
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak2e1
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspl4 = dotproduct(Q,l4)
      Qspk2 = dotproduct(Q,k2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      QspQ = dotproduct(Q,Q)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspe1 = dotproduct(Q,e1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspl5 = dotproduct(Q,l5)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      acc263(1)=abb263(6)
      acc263(2)=abb263(7)
      acc263(3)=abb263(8)
      acc263(4)=abb263(9)
      acc263(5)=abb263(10)
      acc263(6)=abb263(11)
      acc263(7)=abb263(12)
      acc263(8)=abb263(13)
      acc263(9)=abb263(14)
      acc263(10)=abb263(15)
      acc263(11)=abb263(16)
      acc263(12)=abb263(17)
      acc263(13)=abb263(18)
      acc263(14)=abb263(19)
      acc263(15)=abb263(20)
      acc263(16)=abb263(21)
      acc263(17)=abb263(22)
      acc263(18)=abb263(23)
      acc263(19)=abb263(24)
      acc263(20)=abb263(25)
      acc263(21)=abb263(26)
      acc263(22)=abb263(27)
      acc263(23)=abb263(28)
      acc263(24)=abb263(29)
      acc263(25)=abb263(30)
      acc263(26)=abb263(31)
      acc263(27)=abb263(32)
      acc263(28)=abb263(33)
      acc263(29)=abb263(34)
      acc263(30)=abb263(36)
      acc263(31)=abb263(37)
      acc263(32)=abb263(38)
      acc263(33)=abb263(39)
      acc263(34)=abb263(41)
      acc263(35)=abb263(42)
      acc263(36)=abb263(43)
      acc263(37)=abb263(44)
      acc263(38)=abb263(45)
      acc263(39)=abb263(46)
      acc263(40)=abb263(47)
      acc263(41)=abb263(48)
      acc263(42)=abb263(49)
      acc263(43)=abb263(50)
      acc263(44)=abb263(51)
      acc263(45)=abb263(52)
      acc263(46)=abb263(53)
      acc263(47)=abb263(54)
      acc263(48)=abb263(55)
      acc263(49)=abb263(56)
      acc263(50)=abb263(57)
      acc263(51)=abb263(58)
      acc263(52)=abb263(59)
      acc263(53)=abb263(60)
      acc263(54)=abb263(61)
      acc263(55)=abb263(62)
      acc263(56)=abb263(63)
      acc263(57)=abb263(64)
      acc263(58)=abb263(65)
      acc263(59)=abb263(66)
      acc263(60)=abb263(67)
      acc263(61)=abb263(68)
      acc263(62)=abb263(69)
      acc263(63)=abb263(70)
      acc263(64)=abb263(71)
      acc263(65)=abb263(72)
      acc263(66)=abb263(73)
      acc263(67)=abb263(74)
      acc263(68)=abb263(75)
      acc263(69)=abb263(76)
      acc263(70)=abb263(77)
      acc263(71)=abb263(78)
      acc263(72)=abb263(79)
      acc263(73)=abb263(80)
      acc263(74)=abb263(81)
      acc263(75)=abb263(82)
      acc263(76)=abb263(83)
      acc263(77)=abb263(84)
      acc263(78)=abb263(85)
      acc263(79)=abb263(86)
      acc263(80)=abb263(87)
      acc263(81)=abb263(88)
      acc263(82)=abb263(89)
      acc263(83)=abb263(91)
      acc263(84)=abb263(92)
      acc263(85)=abb263(93)
      acc263(86)=abb263(94)
      acc263(87)=abb263(95)
      acc263(88)=abb263(96)
      acc263(89)=abb263(97)
      acc263(90)=abb263(99)
      acc263(91)=abb263(100)
      acc263(92)=abb263(101)
      acc263(93)=abb263(102)
      acc263(94)=abb263(103)
      acc263(95)=abb263(104)
      acc263(96)=abb263(105)
      acc263(97)=abb263(106)
      acc263(98)=abb263(107)
      acc263(99)=abb263(108)
      acc263(100)=abb263(109)
      acc263(101)=abb263(111)
      acc263(102)=abb263(112)
      acc263(103)=abb263(113)
      acc263(104)=abb263(114)
      acc263(105)=abb263(115)
      acc263(106)=abb263(116)
      acc263(107)=abb263(117)
      acc263(108)=abb263(118)
      acc263(109)=abb263(119)
      acc263(110)=abb263(120)
      acc263(111)=abb263(121)
      acc263(112)=abb263(123)
      acc263(113)=abb263(124)
      acc263(114)=abb263(126)
      acc263(115)=abb263(141)
      acc263(116)=abb263(143)
      acc263(117)=abb263(145)
      acc263(118)=abb263(149)
      acc263(119)=abb263(151)
      acc263(120)=abb263(155)
      acc263(121)=abb263(163)
      acc263(122)=abb263(166)
      acc263(123)=abb263(171)
      acc263(124)=abb263(172)
      acc263(125)=abb263(174)
      acc263(126)=abb263(176)
      acc263(127)=abb263(179)
      acc263(128)=abb263(180)
      acc263(129)=abb263(183)
      acc263(130)=abb263(184)
      acc263(131)=abb263(187)
      acc263(132)=abb263(190)
      acc263(133)=abb263(191)
      acc263(134)=abb263(198)
      acc263(135)=abb263(206)
      acc263(136)=abb263(211)
      acc263(137)=Qspvae1k2*acc263(86)
      acc263(138)=Qspval4e1*acc263(119)
      acc263(139)=Qspvae1l4*acc263(136)
      acc263(140)=-Qspl4*acc263(114)
      acc263(141)=Qspk2*acc263(123)
      acc263(142)=-Qspvae2k2*acc263(26)
      acc263(143)=Qspvae2l5*acc263(135)
      acc263(144)=Qspvae2l4*acc263(127)
      acc263(145)=Qspvak2l4*acc263(58)
      acc263(146)=Qspvak1e2*acc263(56)
      acc263(147)=Qspvae2k1*acc263(95)
      acc263(148)=Qspvak2e2*acc263(23)
      acc263(149)=Qspval5e2*acc263(68)
      acc263(150)=Qspvae2k2*acc263(71)
      acc263(150)=acc263(52)+acc263(150)
      acc263(150)=Qspvae1e2*acc263(150)
      acc263(151)=Qspval5e2*acc263(30)
      acc263(151)=acc263(81)+acc263(151)
      acc263(151)=Qspvae2e1*acc263(151)
      acc263(152)=QspQ*acc263(126)
      acc263(137)=acc263(152)+acc263(151)+acc263(150)+acc263(149)+acc263(148)+a&
      &cc263(147)+acc263(146)+acc263(145)+acc263(144)+acc263(143)+acc263(142)+a&
      &cc263(141)+acc263(140)+acc263(139)+acc263(138)+acc263(14)+acc263(137)
      acc263(137)=QspQ*acc263(137)
      acc263(138)=Qspvae2k2*acc263(72)
      acc263(139)=Qspvae2l5*acc263(125)
      acc263(140)=Qspvae2l4*acc263(120)
      acc263(141)=Qspvae2k1*acc263(103)
      acc263(138)=acc263(141)+acc263(140)+acc263(139)+acc263(16)+acc263(138)
      acc263(138)=Qspval5e2*acc263(138)
      acc263(139)=Qspvae2k2*acc263(36)
      acc263(140)=Qspvae2l4*acc263(129)
      acc263(141)=Qspvak2e2*acc263(45)
      acc263(142)=Qspval5e2*acc263(116)
      acc263(139)=acc263(142)+acc263(141)+acc263(140)+acc263(24)+acc263(139)
      acc263(139)=QspQ*acc263(139)
      acc263(140)=Qspvae2k2*acc263(110)
      acc263(141)=Qspvae2l4*acc263(91)
      acc263(140)=acc263(141)+acc263(42)+acc263(140)
      acc263(140)=Qspvak1e2*acc263(140)
      acc263(141)=Qspvae2k2*acc263(76)
      acc263(142)=Qspvae2k1*acc263(78)
      acc263(141)=acc263(142)+acc263(3)+acc263(141)
      acc263(141)=Qspvak2e2*acc263(141)
      acc263(142)=Qspval4k2*acc263(15)
      acc263(143)=Qspval5l4*acc263(1)
      acc263(144)=Qspl4*acc263(47)
      acc263(145)=Qspval5k2*acc263(32)
      acc263(146)=Qspk2*acc263(11)
      acc263(147)=Qspvae2k2*acc263(13)
      acc263(148)=Qspvae2l5*acc263(121)
      acc263(149)=Qspvae2l4*acc263(43)
      acc263(150)=Qspvak2l4*acc263(48)
      acc263(151)=Qspvae2k1*acc263(31)
      acc263(138)=acc263(139)+acc263(138)+acc263(141)+acc263(151)+acc263(140)+a&
      &cc263(150)+acc263(149)+acc263(148)+acc263(147)+acc263(146)+acc263(145)+a&
      &cc263(144)+acc263(143)+acc263(34)+acc263(142)
      acc263(138)=Qspe1*acc263(138)
      acc263(139)=Qspvak2k1*acc263(70)
      acc263(140)=Qspval4k1*acc263(25)
      acc263(141)=Qspval4k2*acc263(20)
      acc263(142)=Qspl5*acc263(50)
      acc263(143)=Qspvak1k2*acc263(18)
      acc263(144)=Qspval5k1*acc263(80)
      acc263(145)=-Qspl4*acc263(128)
      acc263(146)=Qspval5k2*acc263(39)
      acc263(147)=Qspk2*acc263(8)
      acc263(148)=-Qspvak2l4*acc263(100)
      acc263(149)=-Qspval5k1*acc263(30)
      acc263(149)=acc263(57)+acc263(149)
      acc263(149)=Qspvak1e2*acc263(149)
      acc263(150)=Qspval5k2*acc263(73)
      acc263(150)=acc263(61)+acc263(150)
      acc263(150)=Qspvak2e2*acc263(150)
      acc263(151)=Qspl5*acc263(122)
      acc263(151)=acc263(124)+acc263(151)
      acc263(151)=Qspval5e2*acc263(151)
      acc263(139)=acc263(151)+acc263(150)+acc263(149)+acc263(148)+acc263(147)+a&
      &cc263(146)+acc263(145)+acc263(144)+acc263(143)+acc263(142)+acc263(141)+a&
      &cc263(140)+acc263(28)+acc263(139)
      acc263(139)=Qspvae2e1*acc263(139)
      acc263(140)=Qspvak1l4*acc263(74)
      acc263(141)=Qspval5l4*acc263(10)
      acc263(142)=Qspl5*acc263(113)
      acc263(143)=Qspvak1k2*acc263(19)
      acc263(144)=Qspval5k1*acc263(37)
      acc263(145)=Qspval5k2*acc263(40)
      acc263(146)=Qspk2*acc263(66)
      acc263(147)=Qspk2*acc263(38)
      acc263(147)=acc263(51)+acc263(147)
      acc263(147)=Qspvae2k2*acc263(147)
      acc263(148)=Qspval5k2*acc263(71)
      acc263(148)=acc263(75)+acc263(148)
      acc263(148)=Qspvae2l5*acc263(148)
      acc263(149)=Qspvae2l4*acc263(130)
      acc263(150)=Qspvak2l4*acc263(117)
      acc263(151)=-Qspvak1k2*acc263(71)
      acc263(151)=acc263(97)+acc263(151)
      acc263(151)=Qspvae2k1*acc263(151)
      acc263(140)=acc263(151)+acc263(150)+acc263(149)+acc263(148)+acc263(147)+a&
      &cc263(146)+acc263(145)+acc263(144)+acc263(143)+acc263(142)+acc263(141)+a&
      &cc263(79)+acc263(140)
      acc263(140)=Qspvae1e2*acc263(140)
      acc263(141)=Qspvae1k2*acc263(87)
      acc263(142)=-Qspvae1l4*acc263(108)
      acc263(143)=Qspval5k2*acc263(77)
      acc263(144)=Qspk2*acc263(9)
      acc263(145)=Qspvae2l5*acc263(69)
      acc263(146)=Qspvae2l4*acc263(60)
      acc263(147)=Qspvak2l4*acc263(107)
      acc263(148)=Qspvae2k1*acc263(35)
      acc263(141)=acc263(148)+acc263(147)+acc263(146)+acc263(145)+acc263(144)+a&
      &cc263(143)+acc263(142)+acc263(7)+acc263(141)
      acc263(141)=Qspvak2e2*acc263(141)
      acc263(142)=Qspval5k1*acc263(82)
      acc263(143)=Qspvae1k2*acc263(33)
      acc263(144)=Qspvae1l4*acc263(109)
      acc263(145)=Qspk2*acc263(44)
      acc263(146)=Qspvae2l5*acc263(59)
      acc263(147)=Qspvae2l4*acc263(64)
      acc263(148)=Qspvak2l4*acc263(53)
      acc263(142)=acc263(148)+acc263(147)+acc263(146)+acc263(145)+acc263(144)+a&
      &cc263(143)+acc263(17)+acc263(142)
      acc263(142)=Qspvak1e2*acc263(142)
      acc263(143)=Qspvak2e1*acc263(55)
      acc263(144)=Qspvak1k2*acc263(26)
      acc263(145)=Qspval4e1*acc263(106)
      acc263(146)=Qspl4*acc263(90)
      acc263(147)=Qspvak2l4*acc263(22)
      acc263(148)=Qspvak1e2*acc263(46)
      acc263(143)=acc263(148)+acc263(147)+acc263(146)+acc263(145)+acc263(144)+a&
      &cc263(54)+acc263(143)
      acc263(143)=Qspvae2k1*acc263(143)
      acc263(144)=Qspl5*acc263(27)
      acc263(145)=Qspk2*acc263(62)
      acc263(146)=Qspvae2l4*acc263(131)
      acc263(147)=Qspvak2l4*acc263(102)
      acc263(148)=Qspvae2k1*acc263(99)
      acc263(144)=acc263(148)+acc263(147)+acc263(146)+acc263(145)+acc263(12)+ac&
      &c263(144)
      acc263(144)=Qspval5e2*acc263(144)
      acc263(145)=Qspvak2e1*acc263(94)
      acc263(146)=Qspval4e1*acc263(101)
      acc263(147)=Qspvae1l4*acc263(93)
      acc263(148)=Qspvae2l5*acc263(96)
      acc263(145)=acc263(148)+acc263(147)+acc263(146)+acc263(2)+acc263(145)
      acc263(145)=Qspvak2l4*acc263(145)
      acc263(146)=-Qspvae1k2*acc263(118)
      acc263(147)=-Qspvae1l4*acc263(134)
      acc263(146)=acc263(147)+acc263(112)+acc263(146)
      acc263(146)=Qspl4*acc263(146)
      acc263(147)=-Qspl4*acc263(132)
      acc263(148)=-Qspval5k2*acc263(26)
      acc263(147)=acc263(148)+acc263(115)+acc263(147)
      acc263(147)=Qspvae2l5*acc263(147)
      acc263(148)=Qspvak2e1*acc263(92)
      acc263(149)=Qspval4e1*acc263(105)
      acc263(148)=acc263(149)+acc263(89)+acc263(148)
      acc263(148)=Qspvae2l4*acc263(148)
      acc263(149)=Qspvak1l4*acc263(104)
      acc263(150)=Qspvak2k1*acc263(111)
      acc263(151)=Qspval4k1*acc263(88)
      acc263(152)=Qspval4k2*acc263(85)
      acc263(153)=Qspval5l4*acc263(83)
      acc263(154)=Qspvak2e1*acc263(63)
      acc263(155)=Qspl5*acc263(98)
      acc263(156)=Qspvak1k2*acc263(5)
      acc263(157)=Qspval5k1*acc263(41)
      acc263(158)=Qspvae1k2*acc263(84)
      acc263(159)=Qspval4e1*acc263(67)
      acc263(160)=Qspvae1l4*acc263(133)
      acc263(161)=Qspval5k2*acc263(4)
      acc263(162)=-Qspval4e1*acc263(65)
      acc263(162)=acc263(6)+acc263(162)
      acc263(162)=Qspk2*acc263(162)
      acc263(163)=-Qspk2*acc263(29)
      acc263(163)=acc263(49)+acc263(163)
      acc263(163)=Qspvae2k2*acc263(163)
      brack=acc263(21)+acc263(137)+acc263(138)+acc263(139)+acc263(140)+acc263(1&
      &41)+acc263(142)+acc263(143)+acc263(144)+acc263(145)+acc263(146)+acc263(1&
      &47)+acc263(148)+acc263(149)+acc263(150)+acc263(151)+acc263(152)+acc263(1&
      &53)+acc263(154)+acc263(155)+acc263(156)+acc263(157)+acc263(158)+acc263(1&
      &59)+acc263(160)+acc263(161)+acc263(162)+acc263(163)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d263h4l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd263h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d263
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(+Q_ext(0:3),  ki_nin), aimag(+Q_ext(0:3)), ki)
      d263 = 0.0_ki
      d263 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d263, ki), aimag(d263), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d263h4l1_qp
