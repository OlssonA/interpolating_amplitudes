module     p2_gg_httbar_d88h0l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d88h0l132_qp.f90
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
      use p2_gg_httbar_abbrevd88h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(14) :: acd88
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd88(1)=dotproduct(e1,ninjaE3)
      acd88(2)=dotproduct(e2,ninjaE3)
      acd88(3)=dotproduct(ninjaE3,spval4k2)
      acd88(4)=abb88(16)
      acd88(5)=dotproduct(ninjaE3,spval5k2)
      acd88(6)=abb88(81)
      acd88(7)=dotproduct(ninjaE3,spval4l3)
      acd88(8)=abb88(88)
      acd88(9)=dotproduct(ninjaE3,spval3k2)
      acd88(10)=abb88(99)
      acd88(11)=acd88(4)*acd88(3)
      acd88(12)=acd88(6)*acd88(5)
      acd88(13)=acd88(8)*acd88(7)
      acd88(14)=acd88(10)*acd88(9)
      acd88(11)=acd88(14)+acd88(13)+acd88(11)+acd88(12)
      acd88(11)=acd88(11)*acd88(2)*acd88(1)
      brack(ninjaidxt1x0mu0)=acd88(11)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd88h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(136) :: acd88
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd88(1)=dotproduct(e1,ninjaA1)
      acd88(2)=dotproduct(e2,ninjaE3)
      acd88(3)=dotproduct(ninjaE3,spval4k2)
      acd88(4)=abb88(16)
      acd88(5)=dotproduct(ninjaE3,spval3k2)
      acd88(6)=abb88(99)
      acd88(7)=dotproduct(ninjaE3,spval4l3)
      acd88(8)=abb88(88)
      acd88(9)=dotproduct(ninjaE3,spval5k2)
      acd88(10)=abb88(81)
      acd88(11)=dotproduct(e1,ninjaE3)
      acd88(12)=dotproduct(e2,ninjaA1)
      acd88(13)=dotproduct(ninjaA1,spval4k2)
      acd88(14)=dotproduct(ninjaA1,spval3k2)
      acd88(15)=dotproduct(ninjaA1,spval4l3)
      acd88(16)=dotproduct(ninjaA1,spval5k2)
      acd88(17)=dotproduct(k1,ninjaE3)
      acd88(18)=dotproduct(ninjaE3,spvae2e1)
      acd88(19)=abb88(20)
      acd88(20)=dotproduct(ninjaE3,spvae1e2)
      acd88(21)=abb88(32)
      acd88(22)=dotproduct(k2,ninjaE3)
      acd88(23)=abb88(13)
      acd88(24)=abb88(63)
      acd88(25)=abb88(38)
      acd88(26)=abb88(58)
      acd88(27)=abb88(64)
      acd88(28)=dotproduct(l4,ninjaE3)
      acd88(29)=abb88(14)
      acd88(30)=abb88(54)
      acd88(31)=abb88(79)
      acd88(32)=abb88(89)
      acd88(33)=dotproduct(e1,ninjaA0)
      acd88(34)=dotproduct(e2,ninjaA0)
      acd88(35)=dotproduct(ninjaA0,spval4k2)
      acd88(36)=dotproduct(ninjaA0,spval3k2)
      acd88(37)=dotproduct(ninjaA0,spval4l3)
      acd88(38)=dotproduct(ninjaA0,spval5k2)
      acd88(39)=abb88(9)
      acd88(40)=dotproduct(ninjaA0,ninjaE3)
      acd88(41)=abb88(12)
      acd88(42)=abb88(47)
      acd88(43)=dotproduct(ninjaE3,spvae2k2)
      acd88(44)=abb88(18)
      acd88(45)=dotproduct(ninjaE3,spvae2k1)
      acd88(46)=abb88(19)
      acd88(47)=dotproduct(ninjaE3,spvak1k2)
      acd88(48)=abb88(25)
      acd88(49)=dotproduct(ninjaE3,spval5e2)
      acd88(50)=abb88(24)
      acd88(51)=dotproduct(ninjaE3,spval3k1)
      acd88(52)=abb88(28)
      acd88(53)=dotproduct(ninjaE3,spvak2l3)
      acd88(54)=abb88(91)
      acd88(55)=dotproduct(ninjaE3,spval4e2)
      acd88(56)=abb88(37)
      acd88(57)=dotproduct(ninjaE3,spvak1e2)
      acd88(58)=abb88(43)
      acd88(59)=dotproduct(ninjaE3,spvae2l3)
      acd88(60)=abb88(48)
      acd88(61)=dotproduct(ninjaE3,spval3e2)
      acd88(62)=abb88(52)
      acd88(63)=dotproduct(ninjaE3,spval4k1)
      acd88(64)=abb88(59)
      acd88(65)=abb88(86)
      acd88(66)=dotproduct(ninjaE3,spvak1l3)
      acd88(67)=abb88(69)
      acd88(68)=dotproduct(ninjaE3,spvak2e2)
      acd88(69)=abb88(71)
      acd88(70)=dotproduct(ninjaE3,spval5l4)
      acd88(71)=abb88(78)
      acd88(72)=dotproduct(ninjaE3,spval5k1)
      acd88(73)=abb88(85)
      acd88(74)=dotproduct(ninjaE3,spval3l4)
      acd88(75)=abb88(93)
      acd88(76)=abb88(10)
      acd88(77)=abb88(72)
      acd88(78)=abb88(23)
      acd88(79)=abb88(84)
      acd88(80)=abb88(29)
      acd88(81)=dotproduct(ninjaE3,spvae1l3)
      acd88(82)=abb88(30)
      acd88(83)=dotproduct(ninjaE3,spval3e1)
      acd88(84)=abb88(31)
      acd88(85)=dotproduct(ninjaE3,spvae1k2)
      acd88(86)=abb88(33)
      acd88(87)=dotproduct(ninjaE3,spval5e1)
      acd88(88)=abb88(35)
      acd88(89)=dotproduct(ninjaE3,spval4e1)
      acd88(90)=abb88(39)
      acd88(91)=dotproduct(ninjaE3,spvak2e1)
      acd88(92)=abb88(42)
      acd88(93)=abb88(56)
      acd88(94)=abb88(76)
      acd88(95)=abb88(96)
      acd88(96)=dotproduct(ninjaE3,spvae1k1)
      acd88(97)=abb88(65)
      acd88(98)=dotproduct(ninjaE3,spvak1e1)
      acd88(99)=abb88(73)
      acd88(100)=abb88(75)
      acd88(101)=abb88(83)
      acd88(102)=abb88(92)
      acd88(103)=abb88(87)
      acd88(104)=abb88(17)
      acd88(105)=abb88(22)
      acd88(106)=abb88(94)
      acd88(107)=abb88(60)
      acd88(108)=abb88(82)
      acd88(109)=abb88(46)
      acd88(110)=acd88(13)*acd88(4)
      acd88(111)=acd88(14)*acd88(6)
      acd88(112)=acd88(15)*acd88(8)
      acd88(113)=acd88(16)*acd88(10)
      acd88(110)=acd88(113)+acd88(112)+acd88(111)+acd88(110)
      acd88(111)=acd88(11)*acd88(2)
      acd88(110)=acd88(111)*acd88(110)
      acd88(112)=acd88(4)*acd88(3)
      acd88(113)=acd88(6)*acd88(5)
      acd88(114)=acd88(8)*acd88(7)
      acd88(115)=acd88(10)*acd88(9)
      acd88(112)=acd88(115)+acd88(112)+acd88(113)+acd88(114)
      acd88(113)=acd88(112)*acd88(2)
      acd88(114)=acd88(1)*acd88(113)
      acd88(112)=acd88(112)*acd88(11)
      acd88(115)=acd88(12)*acd88(112)
      acd88(110)=acd88(114)+acd88(115)+acd88(110)
      acd88(114)=acd88(24)*acd88(22)
      acd88(115)=acd88(30)*acd88(28)
      acd88(116)=2.0_ki*acd88(40)
      acd88(117)=acd88(76)*acd88(116)
      acd88(118)=acd88(77)*acd88(3)
      acd88(119)=acd88(78)*acd88(47)
      acd88(120)=acd88(79)*acd88(51)
      acd88(121)=acd88(80)*acd88(53)
      acd88(122)=acd88(82)*acd88(81)
      acd88(123)=acd88(84)*acd88(83)
      acd88(124)=acd88(86)*acd88(85)
      acd88(125)=acd88(88)*acd88(87)
      acd88(126)=acd88(90)*acd88(89)
      acd88(127)=acd88(92)*acd88(91)
      acd88(128)=acd88(93)*acd88(63)
      acd88(129)=acd88(94)*acd88(7)
      acd88(130)=-acd88(95)*acd88(66)
      acd88(131)=acd88(97)*acd88(96)
      acd88(132)=acd88(99)*acd88(98)
      acd88(133)=acd88(100)*acd88(70)
      acd88(134)=acd88(101)*acd88(72)
      acd88(135)=acd88(102)*acd88(74)
      acd88(114)=acd88(135)+acd88(134)+acd88(133)+acd88(132)+acd88(131)+acd88(1&
      &30)+acd88(129)+acd88(128)+acd88(127)+acd88(126)+acd88(125)+acd88(124)+ac&
      &d88(123)+acd88(122)+acd88(121)+acd88(120)+acd88(119)+acd88(118)+acd88(11&
      &7)+acd88(115)+acd88(114)
      acd88(114)=acd88(2)*acd88(114)
      acd88(115)=acd88(23)*acd88(22)
      acd88(117)=acd88(29)*acd88(28)
      acd88(118)=acd88(41)*acd88(116)
      acd88(119)=acd88(42)*acd88(3)
      acd88(120)=acd88(44)*acd88(43)
      acd88(121)=acd88(46)*acd88(45)
      acd88(122)=acd88(48)*acd88(47)
      acd88(123)=acd88(50)*acd88(49)
      acd88(124)=acd88(52)*acd88(51)
      acd88(125)=acd88(54)*acd88(53)
      acd88(126)=acd88(56)*acd88(55)
      acd88(127)=acd88(58)*acd88(57)
      acd88(128)=acd88(60)*acd88(59)
      acd88(129)=acd88(62)*acd88(61)
      acd88(130)=acd88(64)*acd88(63)
      acd88(131)=acd88(65)*acd88(7)
      acd88(132)=acd88(67)*acd88(66)
      acd88(133)=-acd88(69)*acd88(68)
      acd88(134)=acd88(71)*acd88(70)
      acd88(135)=acd88(73)*acd88(72)
      acd88(136)=acd88(75)*acd88(74)
      acd88(115)=acd88(136)+acd88(135)+acd88(134)+acd88(133)+acd88(132)+acd88(1&
      &31)+acd88(130)+acd88(129)+acd88(128)+acd88(127)+acd88(126)+acd88(125)+ac&
      &d88(124)+acd88(123)+acd88(122)+acd88(121)+acd88(120)+acd88(119)+acd88(11&
      &8)+acd88(117)+acd88(115)
      acd88(115)=acd88(11)*acd88(115)
      acd88(117)=acd88(35)*acd88(4)
      acd88(118)=acd88(36)*acd88(6)
      acd88(119)=acd88(37)*acd88(8)
      acd88(120)=acd88(38)*acd88(10)
      acd88(117)=acd88(39)+acd88(120)+acd88(119)+acd88(118)+acd88(117)
      acd88(111)=acd88(111)*acd88(117)
      acd88(113)=acd88(33)*acd88(113)
      acd88(112)=acd88(34)*acd88(112)
      acd88(117)=-acd88(25)*acd88(3)
      acd88(118)=acd88(26)*acd88(5)
      acd88(119)=acd88(27)*acd88(9)
      acd88(117)=acd88(119)+acd88(118)+acd88(117)
      acd88(117)=acd88(22)*acd88(117)
      acd88(118)=-acd88(5)*acd88(116)
      acd88(119)=acd88(51)*acd88(47)
      acd88(120)=-acd88(74)*acd88(3)
      acd88(118)=acd88(120)+acd88(118)+acd88(119)
      acd88(118)=acd88(106)*acd88(118)
      acd88(119)=-acd88(9)*acd88(116)
      acd88(120)=-acd88(70)*acd88(3)
      acd88(121)=acd88(72)*acd88(47)
      acd88(119)=acd88(121)+acd88(119)+acd88(120)
      acd88(119)=acd88(108)*acd88(119)
      acd88(120)=acd88(19)*acd88(18)
      acd88(121)=acd88(21)*acd88(20)
      acd88(120)=acd88(121)+acd88(120)
      acd88(121)=acd88(17)-acd88(22)
      acd88(120)=acd88(121)*acd88(120)
      acd88(121)=acd88(31)*acd88(3)
      acd88(122)=acd88(32)*acd88(7)
      acd88(121)=acd88(122)+acd88(121)
      acd88(121)=acd88(28)*acd88(121)
      acd88(122)=acd88(104)*acd88(18)
      acd88(123)=acd88(105)*acd88(20)
      acd88(122)=acd88(123)+acd88(122)
      acd88(122)=acd88(116)*acd88(122)
      acd88(123)=-acd88(3)*acd88(116)
      acd88(124)=acd88(63)*acd88(47)
      acd88(123)=acd88(123)+acd88(124)
      acd88(123)=acd88(103)*acd88(123)
      acd88(116)=-acd88(7)*acd88(116)
      acd88(124)=acd88(66)*acd88(63)
      acd88(116)=acd88(116)+acd88(124)
      acd88(116)=acd88(107)*acd88(116)
      acd88(124)=acd88(109)*acd88(53)*acd88(3)
      acd88(111)=acd88(124)+acd88(116)+acd88(123)+acd88(119)+acd88(118)+acd88(1&
      &13)+acd88(112)+acd88(115)+acd88(114)+acd88(111)+acd88(117)+acd88(122)+ac&
      &d88(121)+acd88(120)
      brack(ninjaidxt0x0mu0)=acd88(111)
      brack(ninjaidxt0x1mu0)=acd88(110)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d88h0_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd88h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d88h0l132_qp
