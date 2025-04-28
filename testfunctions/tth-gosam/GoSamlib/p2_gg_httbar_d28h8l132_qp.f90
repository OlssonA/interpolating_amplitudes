module     p2_gg_httbar_d28h8l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d28h8l132_qp.f90
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
      use p2_gg_httbar_abbrevd28h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(43) :: acd28
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd28(1)=dotproduct(e2,ninjaE3)
      acd28(2)=dotproduct(ninjaE3,spvak2k1)
      acd28(3)=abb28(13)
      acd28(4)=dotproduct(ninjaE3,spvae1l3)
      acd28(5)=abb28(18)
      acd28(6)=dotproduct(ninjaE3,spvak2l4)
      acd28(7)=abb28(29)
      acd28(8)=dotproduct(ninjaE3,spvak2l3)
      acd28(9)=abb28(32)
      acd28(10)=dotproduct(ninjaE3,spvae1l5)
      acd28(11)=abb28(33)
      acd28(12)=dotproduct(ninjaE3,spval3e1)
      acd28(13)=abb28(37)
      acd28(14)=dotproduct(ninjaE3,spvak2e1)
      acd28(15)=abb28(38)
      acd28(16)=dotproduct(ninjaE3,spvak1l5)
      acd28(17)=abb28(39)
      acd28(18)=dotproduct(ninjaE3,spvak1l3)
      acd28(19)=abb28(43)
      acd28(20)=dotproduct(ninjaE3,spval4l3)
      acd28(21)=abb28(62)
      acd28(22)=dotproduct(ninjaE3,spval4l5)
      acd28(23)=abb28(63)
      acd28(24)=dotproduct(ninjaE3,spval3k1)
      acd28(25)=abb28(67)
      acd28(26)=dotproduct(ninjaE3,spval3l4)
      acd28(27)=abb28(74)
      acd28(28)=dotproduct(ninjaE3,spvak2l5)
      acd28(29)=abb28(78)
      acd28(30)=acd28(3)*acd28(2)
      acd28(31)=acd28(5)*acd28(4)
      acd28(32)=acd28(7)*acd28(6)
      acd28(33)=acd28(9)*acd28(8)
      acd28(34)=acd28(11)*acd28(10)
      acd28(35)=acd28(13)*acd28(12)
      acd28(36)=acd28(15)*acd28(14)
      acd28(37)=acd28(17)*acd28(16)
      acd28(38)=acd28(19)*acd28(18)
      acd28(39)=acd28(21)*acd28(20)
      acd28(40)=acd28(23)*acd28(22)
      acd28(41)=acd28(25)*acd28(24)
      acd28(42)=acd28(27)*acd28(26)
      acd28(43)=acd28(29)*acd28(28)
      acd28(30)=acd28(43)+acd28(42)+acd28(41)+acd28(40)+acd28(39)+acd28(38)+acd&
      &28(37)+acd28(36)+acd28(35)+acd28(34)+acd28(33)+acd28(32)+acd28(30)+acd28&
      &(31)
      acd28(30)=acd28(1)*acd28(30)
      brack(ninjaidxt1x0mu0)=acd28(30)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd28h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(160) :: acd28
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd28(1)=dotproduct(e2,ninjaA1)
      acd28(2)=dotproduct(ninjaE3,spvak2k1)
      acd28(3)=abb28(13)
      acd28(4)=dotproduct(ninjaE3,spvak2l3)
      acd28(5)=abb28(32)
      acd28(6)=dotproduct(ninjaE3,spvae1l3)
      acd28(7)=abb28(18)
      acd28(8)=dotproduct(ninjaE3,spvak2l5)
      acd28(9)=abb28(78)
      acd28(10)=dotproduct(ninjaE3,spvak1l5)
      acd28(11)=abb28(39)
      acd28(12)=dotproduct(ninjaE3,spvak2l4)
      acd28(13)=abb28(29)
      acd28(14)=dotproduct(ninjaE3,spvak1l3)
      acd28(15)=abb28(43)
      acd28(16)=dotproduct(ninjaE3,spvae1l5)
      acd28(17)=abb28(33)
      acd28(18)=dotproduct(ninjaE3,spvak2e1)
      acd28(19)=abb28(38)
      acd28(20)=dotproduct(ninjaE3,spval3e1)
      acd28(21)=abb28(37)
      acd28(22)=dotproduct(ninjaE3,spval3k1)
      acd28(23)=abb28(67)
      acd28(24)=dotproduct(ninjaE3,spval4l3)
      acd28(25)=abb28(62)
      acd28(26)=dotproduct(ninjaE3,spval4l5)
      acd28(27)=abb28(63)
      acd28(28)=dotproduct(ninjaE3,spval3l4)
      acd28(29)=abb28(74)
      acd28(30)=dotproduct(e2,ninjaE3)
      acd28(31)=dotproduct(ninjaA1,spvak2k1)
      acd28(32)=dotproduct(ninjaA1,spvak2l3)
      acd28(33)=dotproduct(ninjaA1,spvae1l3)
      acd28(34)=dotproduct(ninjaA1,spvak2l5)
      acd28(35)=dotproduct(ninjaA1,spvak1l5)
      acd28(36)=dotproduct(ninjaA1,spvak2l4)
      acd28(37)=dotproduct(ninjaA1,spvak1l3)
      acd28(38)=dotproduct(ninjaA1,spvae1l5)
      acd28(39)=dotproduct(ninjaA1,spvak2e1)
      acd28(40)=dotproduct(ninjaA1,spval3e1)
      acd28(41)=dotproduct(ninjaA1,spval3k1)
      acd28(42)=dotproduct(ninjaA1,spval4l3)
      acd28(43)=dotproduct(ninjaA1,spval4l5)
      acd28(44)=dotproduct(ninjaA1,spval3l4)
      acd28(45)=dotproduct(k1,ninjaE3)
      acd28(46)=abb28(51)
      acd28(47)=dotproduct(k2,ninjaE3)
      acd28(48)=abb28(15)
      acd28(49)=dotproduct(l4,ninjaE3)
      acd28(50)=abb28(17)
      acd28(51)=dotproduct(e2,ninjaA0)
      acd28(52)=dotproduct(ninjaA0,spvak2k1)
      acd28(53)=dotproduct(ninjaA0,spvak2l3)
      acd28(54)=dotproduct(ninjaA0,spvae1l3)
      acd28(55)=dotproduct(ninjaA0,spvak2l5)
      acd28(56)=dotproduct(ninjaA0,spvak1l5)
      acd28(57)=dotproduct(ninjaA0,spvak2l4)
      acd28(58)=dotproduct(ninjaA0,spvak1l3)
      acd28(59)=dotproduct(ninjaA0,spvae1l5)
      acd28(60)=dotproduct(ninjaA0,spvak2e1)
      acd28(61)=dotproduct(ninjaA0,spval3e1)
      acd28(62)=dotproduct(ninjaA0,spval3k1)
      acd28(63)=dotproduct(ninjaA0,spval4l3)
      acd28(64)=dotproduct(ninjaA0,spval4l5)
      acd28(65)=dotproduct(ninjaA0,spval3l4)
      acd28(66)=abb28(9)
      acd28(67)=dotproduct(ninjaA0,ninjaE3)
      acd28(68)=abb28(23)
      acd28(69)=abb28(10)
      acd28(70)=dotproduct(ninjaE3,spvae2l3)
      acd28(71)=abb28(11)
      acd28(72)=dotproduct(ninjaE3,spvak2e2)
      acd28(73)=abb28(12)
      acd28(74)=abb28(16)
      acd28(75)=dotproduct(ninjaE3,spvae1k1)
      acd28(76)=abb28(19)
      acd28(77)=abb28(20)
      acd28(78)=dotproduct(ninjaE3,spvae2l4)
      acd28(79)=abb28(21)
      acd28(80)=abb28(22)
      acd28(81)=dotproduct(ninjaE3,spvae1k2)
      acd28(82)=abb28(24)
      acd28(83)=abb28(25)
      acd28(84)=dotproduct(ninjaE3,spvae2e1)
      acd28(85)=abb28(26)
      acd28(86)=abb28(27)
      acd28(87)=dotproduct(ninjaE3,spvae1e2)
      acd28(88)=abb28(28)
      acd28(89)=dotproduct(ninjaE3,spvak1e1)
      acd28(90)=abb28(30)
      acd28(91)=dotproduct(ninjaE3,spvae2l5)
      acd28(92)=abb28(31)
      acd28(93)=dotproduct(ninjaE3,spval3k2)
      acd28(94)=abb28(34)
      acd28(95)=abb28(35)
      acd28(96)=dotproduct(ninjaE3,spvae2k1)
      acd28(97)=abb28(40)
      acd28(98)=dotproduct(ninjaE3,spvak1l4)
      acd28(99)=abb28(41)
      acd28(100)=dotproduct(ninjaE3,spvak1e2)
      acd28(101)=abb28(42)
      acd28(102)=abb28(44)
      acd28(103)=dotproduct(ninjaE3,spvak1k2)
      acd28(104)=abb28(45)
      acd28(105)=abb28(46)
      acd28(106)=dotproduct(ninjaE3,spvae2k2)
      acd28(107)=abb28(54)
      acd28(108)=dotproduct(ninjaE3,spval4e2)
      acd28(109)=abb28(56)
      acd28(110)=abb28(60)
      acd28(111)=dotproduct(ninjaE3,spvae1l4)
      acd28(112)=abb28(61)
      acd28(113)=dotproduct(ninjaE3,spval4e1)
      acd28(114)=abb28(68)
      acd28(115)=dotproduct(ninjaE3,spval3e2)
      acd28(116)=abb28(69)
      acd28(117)=dotproduct(ninjaE3,spval4k1)
      acd28(118)=abb28(71)
      acd28(119)=dotproduct(ninjaE3,spval4k2)
      acd28(120)=abb28(72)
      acd28(121)=abb28(73)
      acd28(122)=acd28(29)*acd28(28)
      acd28(123)=acd28(27)*acd28(26)
      acd28(124)=acd28(25)*acd28(24)
      acd28(125)=acd28(23)*acd28(22)
      acd28(126)=acd28(21)*acd28(20)
      acd28(127)=acd28(19)*acd28(18)
      acd28(128)=acd28(17)*acd28(16)
      acd28(129)=acd28(15)*acd28(14)
      acd28(130)=acd28(13)*acd28(12)
      acd28(131)=acd28(11)*acd28(10)
      acd28(132)=acd28(9)*acd28(8)
      acd28(133)=acd28(7)*acd28(6)
      acd28(134)=acd28(5)*acd28(4)
      acd28(135)=acd28(3)*acd28(2)
      acd28(122)=acd28(128)+acd28(129)+acd28(130)+acd28(131)+acd28(124)+acd28(1&
      &25)+acd28(126)+acd28(127)+acd28(122)+acd28(123)+acd28(132)+acd28(133)+ac&
      &d28(134)+acd28(135)
      acd28(123)=acd28(1)*acd28(122)
      acd28(124)=acd28(29)*acd28(44)
      acd28(125)=acd28(27)*acd28(43)
      acd28(126)=acd28(25)*acd28(42)
      acd28(127)=acd28(23)*acd28(41)
      acd28(128)=acd28(21)*acd28(40)
      acd28(129)=acd28(19)*acd28(39)
      acd28(130)=acd28(17)*acd28(38)
      acd28(131)=acd28(15)*acd28(37)
      acd28(132)=acd28(13)*acd28(36)
      acd28(133)=acd28(11)*acd28(35)
      acd28(134)=acd28(9)*acd28(34)
      acd28(135)=acd28(7)*acd28(33)
      acd28(136)=acd28(5)*acd28(32)
      acd28(137)=acd28(3)*acd28(31)
      acd28(124)=acd28(137)+acd28(136)+acd28(135)+acd28(134)+acd28(133)+acd28(1&
      &32)+acd28(131)+acd28(130)+acd28(129)+acd28(128)+acd28(127)+acd28(126)+ac&
      &d28(124)+acd28(125)
      acd28(124)=acd28(30)*acd28(124)
      acd28(123)=acd28(123)+acd28(124)
      acd28(122)=acd28(51)*acd28(122)
      acd28(124)=acd28(29)*acd28(65)
      acd28(125)=acd28(27)*acd28(64)
      acd28(126)=acd28(25)*acd28(63)
      acd28(127)=acd28(23)*acd28(62)
      acd28(128)=acd28(21)*acd28(61)
      acd28(129)=acd28(19)*acd28(60)
      acd28(130)=acd28(17)*acd28(59)
      acd28(131)=acd28(15)*acd28(58)
      acd28(132)=acd28(13)*acd28(57)
      acd28(133)=acd28(11)*acd28(56)
      acd28(134)=acd28(9)*acd28(55)
      acd28(135)=acd28(7)*acd28(54)
      acd28(136)=acd28(5)*acd28(53)
      acd28(137)=acd28(3)*acd28(52)
      acd28(124)=acd28(137)+acd28(136)+acd28(135)+acd28(134)+acd28(133)+acd28(1&
      &32)+acd28(131)+acd28(130)+acd28(129)+acd28(128)+acd28(127)+acd28(126)+ac&
      &d28(125)+acd28(66)+acd28(124)
      acd28(124)=acd28(30)*acd28(124)
      acd28(125)=acd28(119)*acd28(120)
      acd28(126)=acd28(117)*acd28(118)
      acd28(127)=acd28(115)*acd28(116)
      acd28(128)=acd28(113)*acd28(114)
      acd28(129)=acd28(111)*acd28(112)
      acd28(130)=acd28(108)*acd28(109)
      acd28(131)=acd28(106)*acd28(107)
      acd28(132)=acd28(103)*acd28(104)
      acd28(133)=acd28(100)*acd28(101)
      acd28(134)=acd28(98)*acd28(99)
      acd28(135)=acd28(96)*acd28(97)
      acd28(136)=acd28(93)*acd28(94)
      acd28(137)=acd28(91)*acd28(92)
      acd28(138)=acd28(89)*acd28(90)
      acd28(139)=acd28(87)*acd28(88)
      acd28(140)=acd28(84)*acd28(85)
      acd28(141)=acd28(81)*acd28(82)
      acd28(142)=acd28(78)*acd28(79)
      acd28(143)=acd28(75)*acd28(76)
      acd28(144)=acd28(72)*acd28(73)
      acd28(145)=acd28(70)*acd28(71)
      acd28(146)=acd28(67)*acd28(68)
      acd28(147)=acd28(49)*acd28(50)
      acd28(148)=acd28(47)*acd28(48)
      acd28(149)=acd28(45)*acd28(46)
      acd28(150)=acd28(28)*acd28(121)
      acd28(151)=acd28(26)*acd28(110)
      acd28(152)=acd28(24)*acd28(105)
      acd28(153)=acd28(22)*acd28(102)
      acd28(154)=acd28(18)*acd28(95)
      acd28(155)=acd28(14)*acd28(86)
      acd28(156)=acd28(12)*acd28(83)
      acd28(157)=acd28(10)*acd28(80)
      acd28(158)=acd28(8)*acd28(77)
      acd28(159)=acd28(4)*acd28(74)
      acd28(160)=acd28(2)*acd28(69)
      acd28(122)=acd28(124)+acd28(122)+acd28(160)+acd28(159)+acd28(158)+acd28(1&
      &57)+acd28(156)+acd28(155)+acd28(154)+acd28(153)+acd28(152)+acd28(151)+ac&
      &d28(150)+acd28(149)+acd28(148)+acd28(147)-2.0_ki*acd28(146)+acd28(145)+a&
      &cd28(144)+acd28(143)+acd28(142)+acd28(141)+acd28(140)+acd28(139)+acd28(1&
      &38)+acd28(137)+acd28(136)+acd28(135)+acd28(134)+acd28(133)+acd28(132)+ac&
      &d28(131)+acd28(130)+acd28(129)+acd28(128)+acd28(127)+acd28(125)+acd28(12&
      &6)
      brack(ninjaidxt0x0mu0)=acd28(122)
      brack(ninjaidxt0x1mu0)=acd28(123)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d28h8_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd28h8_qp
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
end module     p2_gg_httbar_d28h8l132_qp
