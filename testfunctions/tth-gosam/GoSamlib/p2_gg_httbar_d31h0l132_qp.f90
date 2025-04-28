module     p2_gg_httbar_d31h0l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d31h0l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1x0mu0 = 0
   integer, parameter :: ninjaidxt0x0mu0 = 1
   integer, parameter :: ninjaidxt0x1mu0 = 2
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd31h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(43) :: acd31
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd31(1)=dotproduct(k2,ninjaE3)
      acd31(2)=dotproduct(e2,ninjaE3)
      acd31(3)=abb31(43)
      acd31(4)=dotproduct(ninjaE3,spval3k1)
      acd31(5)=abb31(17)
      acd31(6)=dotproduct(ninjaE3,spvak2l3)
      acd31(7)=abb31(28)
      acd31(8)=dotproduct(ninjaE3,spvak1l3)
      acd31(9)=abb31(32)
      acd31(10)=dotproduct(ninjaE3,spvae1k2)
      acd31(11)=abb31(33)
      acd31(12)=dotproduct(ninjaE3,spvak1k2)
      acd31(13)=abb31(40)
      acd31(14)=dotproduct(ninjaE3,spval4k1)
      acd31(15)=abb31(51)
      acd31(16)=dotproduct(ninjaE3,spval3l5)
      acd31(17)=abb31(52)
      acd31(18)=dotproduct(ninjaE3,spval5k2)
      acd31(19)=abb31(53)
      acd31(20)=dotproduct(ninjaE3,spval4e1)
      acd31(21)=abb31(60)
      acd31(22)=dotproduct(ninjaE3,spval3e1)
      acd31(23)=abb31(61)
      acd31(24)=dotproduct(ninjaE3,spvae1l3)
      acd31(25)=abb31(68)
      acd31(26)=dotproduct(ninjaE3,spval5l3)
      acd31(27)=abb31(69)
      acd31(28)=dotproduct(ninjaE3,spval4l5)
      acd31(29)=abb31(71)
      acd31(30)=acd31(3)*acd31(1)
      acd31(31)=acd31(5)*acd31(4)
      acd31(32)=acd31(7)*acd31(6)
      acd31(33)=acd31(9)*acd31(8)
      acd31(34)=acd31(11)*acd31(10)
      acd31(35)=acd31(13)*acd31(12)
      acd31(36)=acd31(15)*acd31(14)
      acd31(37)=acd31(17)*acd31(16)
      acd31(38)=acd31(19)*acd31(18)
      acd31(39)=acd31(21)*acd31(20)
      acd31(40)=acd31(23)*acd31(22)
      acd31(41)=acd31(25)*acd31(24)
      acd31(42)=acd31(27)*acd31(26)
      acd31(43)=acd31(29)*acd31(28)
      acd31(30)=acd31(43)+acd31(42)+acd31(41)+acd31(40)+acd31(39)+acd31(38)+acd&
      &31(37)+acd31(36)+acd31(35)+acd31(34)+acd31(33)+acd31(32)+acd31(30)+acd31&
      &(31)
      acd31(30)=acd31(2)*acd31(30)
      brack(ninjaidxt1x0mu0)=acd31(30)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd31h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(160) :: acd31
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd31(1)=dotproduct(k2,ninjaA1)
      acd31(2)=dotproduct(e2,ninjaE3)
      acd31(3)=abb31(43)
      acd31(4)=dotproduct(k2,ninjaE3)
      acd31(5)=dotproduct(e2,ninjaA1)
      acd31(6)=dotproduct(ninjaE3,spvak2l3)
      acd31(7)=abb31(28)
      acd31(8)=dotproduct(ninjaE3,spval3k1)
      acd31(9)=abb31(17)
      acd31(10)=dotproduct(ninjaE3,spvak1l3)
      acd31(11)=abb31(32)
      acd31(12)=dotproduct(ninjaE3,spval3l5)
      acd31(13)=abb31(52)
      acd31(14)=dotproduct(ninjaE3,spval5k2)
      acd31(15)=abb31(53)
      acd31(16)=dotproduct(ninjaE3,spvae1k2)
      acd31(17)=abb31(33)
      acd31(18)=dotproduct(ninjaE3,spvak1k2)
      acd31(19)=abb31(40)
      acd31(20)=dotproduct(ninjaE3,spval4l5)
      acd31(21)=abb31(71)
      acd31(22)=dotproduct(ninjaE3,spval4k1)
      acd31(23)=abb31(51)
      acd31(24)=dotproduct(ninjaE3,spval5l3)
      acd31(25)=abb31(69)
      acd31(26)=dotproduct(ninjaE3,spval4e1)
      acd31(27)=abb31(60)
      acd31(28)=dotproduct(ninjaE3,spval3e1)
      acd31(29)=abb31(61)
      acd31(30)=dotproduct(ninjaE3,spvae1l3)
      acd31(31)=abb31(68)
      acd31(32)=dotproduct(ninjaA1,spvak2l3)
      acd31(33)=dotproduct(ninjaA1,spval3k1)
      acd31(34)=dotproduct(ninjaA1,spvak1l3)
      acd31(35)=dotproduct(ninjaA1,spval3l5)
      acd31(36)=dotproduct(ninjaA1,spval5k2)
      acd31(37)=dotproduct(ninjaA1,spvae1k2)
      acd31(38)=dotproduct(ninjaA1,spvak1k2)
      acd31(39)=dotproduct(ninjaA1,spval4l5)
      acd31(40)=dotproduct(ninjaA1,spval4k1)
      acd31(41)=dotproduct(ninjaA1,spval5l3)
      acd31(42)=dotproduct(ninjaA1,spval4e1)
      acd31(43)=dotproduct(ninjaA1,spval3e1)
      acd31(44)=dotproduct(ninjaA1,spvae1l3)
      acd31(45)=dotproduct(k1,ninjaE3)
      acd31(46)=abb31(19)
      acd31(47)=dotproduct(k2,ninjaA0)
      acd31(48)=dotproduct(e2,ninjaA0)
      acd31(49)=abb31(13)
      acd31(50)=dotproduct(l5,ninjaE3)
      acd31(51)=abb31(23)
      acd31(52)=dotproduct(ninjaA0,spvak2l3)
      acd31(53)=dotproduct(ninjaA0,spval3k1)
      acd31(54)=dotproduct(ninjaA0,spvak1l3)
      acd31(55)=dotproduct(ninjaA0,spval3l5)
      acd31(56)=dotproduct(ninjaA0,spval5k2)
      acd31(57)=dotproduct(ninjaA0,spvae1k2)
      acd31(58)=dotproduct(ninjaA0,spvak1k2)
      acd31(59)=dotproduct(ninjaA0,spval4l5)
      acd31(60)=dotproduct(ninjaA0,spval4k1)
      acd31(61)=dotproduct(ninjaA0,spval5l3)
      acd31(62)=dotproduct(ninjaA0,spval4e1)
      acd31(63)=dotproduct(ninjaA0,spval3e1)
      acd31(64)=dotproduct(ninjaA0,spvae1l3)
      acd31(65)=abb31(27)
      acd31(66)=dotproduct(ninjaA0,ninjaE3)
      acd31(67)=abb31(25)
      acd31(68)=dotproduct(ninjaE3,spvak2l5)
      acd31(69)=abb31(9)
      acd31(70)=dotproduct(ninjaE3,spvae2l5)
      acd31(71)=abb31(10)
      acd31(72)=dotproduct(ninjaE3,spvak2e2)
      acd31(73)=abb31(11)
      acd31(74)=dotproduct(ninjaE3,spvae2k2)
      acd31(75)=abb31(12)
      acd31(76)=abb31(14)
      acd31(77)=dotproduct(ninjaE3,spval3k2)
      acd31(78)=abb31(15)
      acd31(79)=abb31(18)
      acd31(80)=dotproduct(ninjaE3,spvak2k1)
      acd31(81)=abb31(20)
      acd31(82)=dotproduct(ninjaE3,spvak1l5)
      acd31(83)=abb31(21)
      acd31(84)=abb31(22)
      acd31(85)=abb31(24)
      acd31(86)=abb31(26)
      acd31(87)=dotproduct(ninjaE3,spvae1e2)
      acd31(88)=abb31(29)
      acd31(89)=abb31(30)
      acd31(90)=dotproduct(ninjaE3,spvae1k1)
      acd31(91)=abb31(31)
      acd31(92)=dotproduct(ninjaE3,spvak1e1)
      acd31(93)=abb31(34)
      acd31(94)=dotproduct(ninjaE3,spval5e2)
      acd31(95)=abb31(36)
      acd31(96)=dotproduct(ninjaE3,spvae1l5)
      acd31(97)=abb31(37)
      acd31(98)=abb31(38)
      acd31(99)=dotproduct(ninjaE3,spval4k2)
      acd31(100)=abb31(39)
      acd31(101)=dotproduct(ninjaE3,spvae2l3)
      acd31(102)=abb31(41)
      acd31(103)=abb31(42)
      acd31(104)=dotproduct(ninjaE3,spvae2e1)
      acd31(105)=abb31(45)
      acd31(106)=dotproduct(ninjaE3,spval5e1)
      acd31(107)=abb31(47)
      acd31(108)=dotproduct(ninjaE3,spval4e2)
      acd31(109)=abb31(49)
      acd31(110)=dotproduct(ninjaE3,spvak2e1)
      acd31(111)=abb31(50)
      acd31(112)=abb31(54)
      acd31(113)=dotproduct(ninjaE3,spvae2k1)
      acd31(114)=abb31(55)
      acd31(115)=dotproduct(ninjaE3,spval5k1)
      acd31(116)=abb31(56)
      acd31(117)=dotproduct(ninjaE3,spvak1e2)
      acd31(118)=abb31(58)
      acd31(119)=abb31(59)
      acd31(120)=dotproduct(ninjaE3,spval3e2)
      acd31(121)=abb31(62)
      acd31(122)=acd31(31)*acd31(30)
      acd31(123)=acd31(29)*acd31(28)
      acd31(124)=acd31(27)*acd31(26)
      acd31(125)=acd31(25)*acd31(24)
      acd31(126)=acd31(23)*acd31(22)
      acd31(127)=acd31(21)*acd31(20)
      acd31(128)=acd31(19)*acd31(18)
      acd31(129)=acd31(17)*acd31(16)
      acd31(130)=acd31(15)*acd31(14)
      acd31(131)=acd31(13)*acd31(12)
      acd31(132)=acd31(11)*acd31(10)
      acd31(133)=acd31(9)*acd31(8)
      acd31(134)=acd31(7)*acd31(6)
      acd31(135)=acd31(3)*acd31(4)
      acd31(122)=acd31(128)+acd31(129)+acd31(130)+acd31(131)+acd31(124)+acd31(1&
      &25)+acd31(126)+acd31(127)+acd31(122)+acd31(123)+acd31(132)+acd31(133)+ac&
      &d31(134)+acd31(135)
      acd31(123)=acd31(5)*acd31(122)
      acd31(124)=acd31(31)*acd31(44)
      acd31(125)=acd31(29)*acd31(43)
      acd31(126)=acd31(27)*acd31(42)
      acd31(127)=acd31(25)*acd31(41)
      acd31(128)=acd31(23)*acd31(40)
      acd31(129)=acd31(21)*acd31(39)
      acd31(130)=acd31(19)*acd31(38)
      acd31(131)=acd31(17)*acd31(37)
      acd31(132)=acd31(15)*acd31(36)
      acd31(133)=acd31(13)*acd31(35)
      acd31(134)=acd31(11)*acd31(34)
      acd31(135)=acd31(9)*acd31(33)
      acd31(136)=acd31(7)*acd31(32)
      acd31(137)=acd31(3)*acd31(1)
      acd31(124)=acd31(137)+acd31(136)+acd31(135)+acd31(134)+acd31(133)+acd31(1&
      &32)+acd31(131)+acd31(130)+acd31(129)+acd31(128)+acd31(127)+acd31(126)+ac&
      &d31(124)+acd31(125)
      acd31(124)=acd31(2)*acd31(124)
      acd31(123)=acd31(123)+acd31(124)
      acd31(122)=acd31(48)*acd31(122)
      acd31(124)=acd31(31)*acd31(64)
      acd31(125)=acd31(29)*acd31(63)
      acd31(126)=acd31(27)*acd31(62)
      acd31(127)=acd31(25)*acd31(61)
      acd31(128)=acd31(23)*acd31(60)
      acd31(129)=acd31(21)*acd31(59)
      acd31(130)=acd31(19)*acd31(58)
      acd31(131)=acd31(17)*acd31(57)
      acd31(132)=acd31(15)*acd31(56)
      acd31(133)=acd31(13)*acd31(55)
      acd31(134)=acd31(11)*acd31(54)
      acd31(135)=acd31(9)*acd31(53)
      acd31(136)=acd31(7)*acd31(52)
      acd31(137)=acd31(3)*acd31(47)
      acd31(124)=acd31(137)+acd31(136)+acd31(135)+acd31(134)+acd31(133)+acd31(1&
      &32)+acd31(131)+acd31(130)+acd31(129)+acd31(128)+acd31(127)+acd31(126)+ac&
      &d31(125)+acd31(65)+acd31(124)
      acd31(124)=acd31(2)*acd31(124)
      acd31(125)=acd31(120)*acd31(121)
      acd31(126)=acd31(117)*acd31(118)
      acd31(127)=acd31(115)*acd31(116)
      acd31(128)=acd31(113)*acd31(114)
      acd31(129)=acd31(110)*acd31(111)
      acd31(130)=acd31(108)*acd31(109)
      acd31(131)=acd31(106)*acd31(107)
      acd31(132)=acd31(104)*acd31(105)
      acd31(133)=acd31(101)*acd31(102)
      acd31(134)=acd31(99)*acd31(100)
      acd31(135)=acd31(96)*acd31(97)
      acd31(136)=acd31(94)*acd31(95)
      acd31(137)=acd31(92)*acd31(93)
      acd31(138)=acd31(90)*acd31(91)
      acd31(139)=acd31(87)*acd31(88)
      acd31(140)=acd31(82)*acd31(83)
      acd31(141)=acd31(80)*acd31(81)
      acd31(142)=acd31(77)*acd31(78)
      acd31(143)=acd31(74)*acd31(75)
      acd31(144)=acd31(72)*acd31(73)
      acd31(145)=acd31(70)*acd31(71)
      acd31(146)=acd31(68)*acd31(69)
      acd31(147)=acd31(66)*acd31(67)
      acd31(148)=acd31(50)*acd31(51)
      acd31(149)=acd31(45)*acd31(46)
      acd31(150)=acd31(24)*acd31(119)
      acd31(151)=acd31(22)*acd31(112)
      acd31(152)=acd31(20)*acd31(103)
      acd31(153)=acd31(18)*acd31(98)
      acd31(154)=acd31(16)*acd31(89)
      acd31(155)=acd31(14)*acd31(86)
      acd31(156)=acd31(12)*acd31(85)
      acd31(157)=acd31(10)*acd31(84)
      acd31(158)=acd31(8)*acd31(79)
      acd31(159)=acd31(6)*acd31(76)
      acd31(160)=acd31(4)*acd31(49)
      acd31(122)=acd31(124)+acd31(122)+acd31(160)+acd31(159)+acd31(158)+acd31(1&
      &57)+acd31(156)+acd31(155)+acd31(154)+acd31(153)+acd31(152)+acd31(151)+ac&
      &d31(150)+acd31(149)+acd31(148)+2.0_ki*acd31(147)+acd31(146)+acd31(145)+a&
      &cd31(144)+acd31(143)+acd31(142)+acd31(141)+acd31(140)+acd31(139)+acd31(1&
      &38)+acd31(137)+acd31(136)+acd31(135)+acd31(134)+acd31(133)+acd31(132)+ac&
      &d31(131)+acd31(130)+acd31(129)+acd31(128)+acd31(127)+acd31(125)+acd31(12&
      &6)
      brack(ninjaidxt0x0mu0)=acd31(122)
      brack(ninjaidxt0x1mu0)=acd31(123)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d31h0_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd31h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d31h0l132_qp
