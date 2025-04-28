module     p2_gg_httbar_d82h0l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d82h0l132_qp.f90
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
      use p2_gg_httbar_abbrevd82h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(14) :: acd82
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd82(1)=dotproduct(e1,ninjaE3)
      acd82(2)=dotproduct(e2,ninjaE3)
      acd82(3)=dotproduct(ninjaE3,spval5k2)
      acd82(4)=abb82(16)
      acd82(5)=dotproduct(ninjaE3,spval5l3)
      acd82(6)=abb82(76)
      acd82(7)=dotproduct(ninjaE3,spval4k2)
      acd82(8)=abb82(87)
      acd82(9)=dotproduct(ninjaE3,spval3k2)
      acd82(10)=abb82(99)
      acd82(11)=acd82(4)*acd82(3)
      acd82(12)=acd82(6)*acd82(5)
      acd82(13)=acd82(8)*acd82(7)
      acd82(14)=acd82(10)*acd82(9)
      acd82(11)=acd82(14)+acd82(13)+acd82(11)+acd82(12)
      acd82(11)=acd82(11)*acd82(2)*acd82(1)
      brack(ninjaidxt1x0mu0)=acd82(11)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd82h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(136) :: acd82
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd82(1)=dotproduct(e1,ninjaA1)
      acd82(2)=dotproduct(e2,ninjaE3)
      acd82(3)=dotproduct(ninjaE3,spval5k2)
      acd82(4)=abb82(16)
      acd82(5)=dotproduct(ninjaE3,spval3k2)
      acd82(6)=abb82(99)
      acd82(7)=dotproduct(ninjaE3,spval5l3)
      acd82(8)=abb82(76)
      acd82(9)=dotproduct(ninjaE3,spval4k2)
      acd82(10)=abb82(87)
      acd82(11)=dotproduct(e1,ninjaE3)
      acd82(12)=dotproduct(e2,ninjaA1)
      acd82(13)=dotproduct(ninjaA1,spval5k2)
      acd82(14)=dotproduct(ninjaA1,spval3k2)
      acd82(15)=dotproduct(ninjaA1,spval5l3)
      acd82(16)=dotproduct(ninjaA1,spval4k2)
      acd82(17)=dotproduct(k1,ninjaE3)
      acd82(18)=dotproduct(ninjaE3,spvae2e1)
      acd82(19)=abb82(20)
      acd82(20)=dotproduct(ninjaE3,spvae1e2)
      acd82(21)=abb82(32)
      acd82(22)=dotproduct(k2,ninjaE3)
      acd82(23)=abb82(13)
      acd82(24)=abb82(63)
      acd82(25)=abb82(38)
      acd82(26)=abb82(58)
      acd82(27)=abb82(64)
      acd82(28)=dotproduct(l5,ninjaE3)
      acd82(29)=abb82(14)
      acd82(30)=abb82(54)
      acd82(31)=abb82(79)
      acd82(32)=abb82(78)
      acd82(33)=dotproduct(e1,ninjaA0)
      acd82(34)=dotproduct(e2,ninjaA0)
      acd82(35)=dotproduct(ninjaA0,spval5k2)
      acd82(36)=dotproduct(ninjaA0,spval3k2)
      acd82(37)=dotproduct(ninjaA0,spval5l3)
      acd82(38)=dotproduct(ninjaA0,spval4k2)
      acd82(39)=abb82(9)
      acd82(40)=dotproduct(ninjaA0,ninjaE3)
      acd82(41)=abb82(12)
      acd82(42)=abb82(47)
      acd82(43)=dotproduct(ninjaE3,spvae2k2)
      acd82(44)=abb82(18)
      acd82(45)=dotproduct(ninjaE3,spvae2k1)
      acd82(46)=abb82(19)
      acd82(47)=dotproduct(ninjaE3,spvak1k2)
      acd82(48)=abb82(25)
      acd82(49)=dotproduct(ninjaE3,spval5e2)
      acd82(50)=abb82(24)
      acd82(51)=dotproduct(ninjaE3,spval3k1)
      acd82(52)=abb82(28)
      acd82(53)=dotproduct(ninjaE3,spvak2l3)
      acd82(54)=abb82(91)
      acd82(55)=dotproduct(ninjaE3,spval4e2)
      acd82(56)=abb82(35)
      acd82(57)=dotproduct(ninjaE3,spvak1e2)
      acd82(58)=abb82(43)
      acd82(59)=dotproduct(ninjaE3,spvae2l3)
      acd82(60)=abb82(48)
      acd82(61)=dotproduct(ninjaE3,spval3e2)
      acd82(62)=abb82(52)
      acd82(63)=dotproduct(ninjaE3,spval5k1)
      acd82(64)=abb82(59)
      acd82(65)=abb82(75)
      acd82(66)=dotproduct(ninjaE3,spvak1l3)
      acd82(67)=abb82(69)
      acd82(68)=dotproduct(ninjaE3,spvak2e2)
      acd82(69)=abb82(71)
      acd82(70)=dotproduct(ninjaE3,spval4k1)
      acd82(71)=abb82(89)
      acd82(72)=dotproduct(ninjaE3,spval4l5)
      acd82(73)=abb82(85)
      acd82(74)=dotproduct(ninjaE3,spval3l5)
      acd82(75)=abb82(93)
      acd82(76)=abb82(10)
      acd82(77)=abb82(72)
      acd82(78)=abb82(23)
      acd82(79)=abb82(88)
      acd82(80)=abb82(29)
      acd82(81)=dotproduct(ninjaE3,spvae1l3)
      acd82(82)=abb82(30)
      acd82(83)=dotproduct(ninjaE3,spval5e1)
      acd82(84)=abb82(31)
      acd82(85)=dotproduct(ninjaE3,spvae1k2)
      acd82(86)=abb82(33)
      acd82(87)=dotproduct(ninjaE3,spval3e1)
      acd82(88)=abb82(37)
      acd82(89)=dotproduct(ninjaE3,spval4e1)
      acd82(90)=abb82(39)
      acd82(91)=dotproduct(ninjaE3,spvak2e1)
      acd82(92)=abb82(42)
      acd82(93)=abb82(56)
      acd82(94)=abb82(74)
      acd82(95)=abb82(96)
      acd82(96)=dotproduct(ninjaE3,spvae1k1)
      acd82(97)=abb82(65)
      acd82(98)=dotproduct(ninjaE3,spvak1e1)
      acd82(99)=abb82(73)
      acd82(100)=abb82(83)
      acd82(101)=abb82(84)
      acd82(102)=abb82(92)
      acd82(103)=abb82(81)
      acd82(104)=abb82(17)
      acd82(105)=abb82(22)
      acd82(106)=abb82(94)
      acd82(107)=abb82(60)
      acd82(108)=abb82(86)
      acd82(109)=abb82(46)
      acd82(110)=acd82(13)*acd82(4)
      acd82(111)=acd82(14)*acd82(6)
      acd82(112)=acd82(15)*acd82(8)
      acd82(113)=acd82(16)*acd82(10)
      acd82(110)=acd82(113)+acd82(112)+acd82(111)+acd82(110)
      acd82(111)=acd82(11)*acd82(2)
      acd82(110)=acd82(111)*acd82(110)
      acd82(112)=acd82(4)*acd82(3)
      acd82(113)=acd82(6)*acd82(5)
      acd82(114)=acd82(8)*acd82(7)
      acd82(115)=acd82(10)*acd82(9)
      acd82(112)=acd82(115)+acd82(112)+acd82(113)+acd82(114)
      acd82(113)=acd82(112)*acd82(2)
      acd82(114)=acd82(1)*acd82(113)
      acd82(112)=acd82(112)*acd82(11)
      acd82(115)=acd82(12)*acd82(112)
      acd82(110)=acd82(114)+acd82(115)+acd82(110)
      acd82(114)=acd82(24)*acd82(22)
      acd82(115)=acd82(30)*acd82(28)
      acd82(116)=2.0_ki*acd82(40)
      acd82(117)=acd82(76)*acd82(116)
      acd82(118)=acd82(77)*acd82(3)
      acd82(119)=acd82(78)*acd82(47)
      acd82(120)=acd82(79)*acd82(51)
      acd82(121)=acd82(80)*acd82(53)
      acd82(122)=acd82(82)*acd82(81)
      acd82(123)=acd82(84)*acd82(83)
      acd82(124)=acd82(86)*acd82(85)
      acd82(125)=acd82(88)*acd82(87)
      acd82(126)=acd82(90)*acd82(89)
      acd82(127)=-acd82(92)*acd82(91)
      acd82(128)=acd82(93)*acd82(63)
      acd82(129)=acd82(94)*acd82(7)
      acd82(130)=acd82(95)*acd82(66)
      acd82(131)=acd82(97)*acd82(96)
      acd82(132)=acd82(99)*acd82(98)
      acd82(133)=acd82(100)*acd82(70)
      acd82(134)=acd82(101)*acd82(72)
      acd82(135)=acd82(102)*acd82(74)
      acd82(114)=acd82(135)+acd82(134)+acd82(133)+acd82(132)+acd82(131)+acd82(1&
      &30)+acd82(129)+acd82(128)+acd82(127)+acd82(126)+acd82(125)+acd82(124)+ac&
      &d82(123)+acd82(122)+acd82(121)+acd82(120)+acd82(119)+acd82(118)+acd82(11&
      &7)+acd82(115)+acd82(114)
      acd82(114)=acd82(2)*acd82(114)
      acd82(115)=acd82(23)*acd82(22)
      acd82(117)=-acd82(29)*acd82(28)
      acd82(118)=acd82(41)*acd82(116)
      acd82(119)=acd82(42)*acd82(3)
      acd82(120)=acd82(44)*acd82(43)
      acd82(121)=acd82(46)*acd82(45)
      acd82(122)=acd82(48)*acd82(47)
      acd82(123)=acd82(50)*acd82(49)
      acd82(124)=acd82(52)*acd82(51)
      acd82(125)=acd82(54)*acd82(53)
      acd82(126)=acd82(56)*acd82(55)
      acd82(127)=acd82(58)*acd82(57)
      acd82(128)=acd82(60)*acd82(59)
      acd82(129)=acd82(62)*acd82(61)
      acd82(130)=acd82(64)*acd82(63)
      acd82(131)=acd82(65)*acd82(7)
      acd82(132)=acd82(67)*acd82(66)
      acd82(133)=acd82(69)*acd82(68)
      acd82(134)=acd82(71)*acd82(70)
      acd82(135)=acd82(73)*acd82(72)
      acd82(136)=acd82(75)*acd82(74)
      acd82(115)=acd82(136)+acd82(135)+acd82(134)+acd82(133)+acd82(132)+acd82(1&
      &31)+acd82(130)+acd82(129)+acd82(128)+acd82(127)+acd82(126)+acd82(125)+ac&
      &d82(124)+acd82(123)+acd82(122)+acd82(121)+acd82(120)+acd82(119)+acd82(11&
      &8)+acd82(117)+acd82(115)
      acd82(115)=acd82(11)*acd82(115)
      acd82(117)=acd82(35)*acd82(4)
      acd82(118)=acd82(36)*acd82(6)
      acd82(119)=acd82(37)*acd82(8)
      acd82(120)=acd82(38)*acd82(10)
      acd82(117)=acd82(39)+acd82(120)+acd82(119)+acd82(118)+acd82(117)
      acd82(111)=acd82(111)*acd82(117)
      acd82(113)=acd82(33)*acd82(113)
      acd82(112)=acd82(34)*acd82(112)
      acd82(117)=acd82(25)*acd82(3)
      acd82(118)=acd82(26)*acd82(5)
      acd82(119)=acd82(27)*acd82(9)
      acd82(117)=acd82(119)+acd82(118)+acd82(117)
      acd82(117)=acd82(22)*acd82(117)
      acd82(118)=acd82(5)*acd82(116)
      acd82(119)=-acd82(51)*acd82(47)
      acd82(120)=acd82(74)*acd82(3)
      acd82(118)=acd82(120)+acd82(118)+acd82(119)
      acd82(118)=acd82(106)*acd82(118)
      acd82(119)=acd82(9)*acd82(116)
      acd82(120)=-acd82(70)*acd82(47)
      acd82(121)=acd82(72)*acd82(3)
      acd82(119)=acd82(121)+acd82(119)+acd82(120)
      acd82(119)=acd82(108)*acd82(119)
      acd82(120)=-acd82(19)*acd82(18)
      acd82(121)=-acd82(21)*acd82(20)
      acd82(120)=acd82(121)+acd82(120)
      acd82(121)=acd82(17)-acd82(22)
      acd82(120)=acd82(121)*acd82(120)
      acd82(121)=acd82(31)*acd82(3)
      acd82(122)=acd82(32)*acd82(7)
      acd82(121)=acd82(122)+acd82(121)
      acd82(121)=acd82(28)*acd82(121)
      acd82(122)=acd82(104)*acd82(18)
      acd82(123)=acd82(105)*acd82(20)
      acd82(122)=acd82(123)+acd82(122)
      acd82(122)=acd82(116)*acd82(122)
      acd82(123)=acd82(3)*acd82(116)
      acd82(124)=-acd82(63)*acd82(47)
      acd82(123)=acd82(123)+acd82(124)
      acd82(123)=acd82(103)*acd82(123)
      acd82(116)=acd82(7)*acd82(116)
      acd82(124)=-acd82(66)*acd82(63)
      acd82(116)=acd82(116)+acd82(124)
      acd82(116)=acd82(107)*acd82(116)
      acd82(124)=acd82(109)*acd82(53)*acd82(3)
      acd82(111)=acd82(124)+acd82(116)+acd82(123)+acd82(119)+acd82(118)+acd82(1&
      &13)+acd82(112)+acd82(115)+acd82(114)+acd82(111)+acd82(117)+acd82(122)+ac&
      &d82(121)+acd82(120)
      brack(ninjaidxt0x0mu0)=acd82(111)
      brack(ninjaidxt0x1mu0)=acd82(110)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d82h0_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd82h0_qp
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
end module     p2_gg_httbar_d82h0l132_qp
