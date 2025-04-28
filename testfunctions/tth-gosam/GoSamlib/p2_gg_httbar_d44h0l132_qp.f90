module     p2_gg_httbar_d44h0l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d44h0l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2x0mu0 = 0
   integer, parameter :: ninjaidxt1x0mu0 = 1
   integer, parameter :: ninjaidxt1x1mu0 = 2
   integer, parameter :: ninjaidxt0x0mu0 = 3
   integer, parameter :: ninjaidxt0x0mu2 = 4
   integer, parameter :: ninjaidxt0x1mu0 = 5
   integer, parameter :: ninjaidxt0x2mu0 = 6
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd44h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(87) :: acd44
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd44(1)=dotproduct(k1,ninjaE3)
      acd44(2)=abb44(16)
      acd44(3)=dotproduct(k2,ninjaE3)
      acd44(4)=abb44(18)
      acd44(5)=dotproduct(l4,ninjaE3)
      acd44(6)=abb44(20)
      acd44(7)=dotproduct(ninjaE3,spvak1k2)
      acd44(8)=abb44(15)
      acd44(9)=dotproduct(ninjaE3,spvae2l4)
      acd44(10)=abb44(17)
      acd44(11)=dotproduct(ninjaE3,spvak1l4)
      acd44(12)=abb44(21)
      acd44(13)=dotproduct(ninjaE3,spvak2l5)
      acd44(14)=abb44(25)
      acd44(15)=dotproduct(ninjaE3,spvak2k1)
      acd44(16)=abb44(26)
      acd44(17)=dotproduct(ninjaE3,spvae2e1)
      acd44(18)=abb44(27)
      acd44(19)=dotproduct(ninjaE3,spvak1l5)
      acd44(20)=abb44(28)
      acd44(21)=dotproduct(ninjaE3,spval5e1)
      acd44(22)=abb44(31)
      acd44(23)=dotproduct(ninjaE3,spval4e1)
      acd44(24)=abb44(32)
      acd44(25)=dotproduct(ninjaE3,spvae1e2)
      acd44(26)=abb44(33)
      acd44(27)=dotproduct(ninjaE3,spvae1k1)
      acd44(28)=abb44(34)
      acd44(29)=dotproduct(ninjaE3,spvae1l4)
      acd44(30)=abb44(35)
      acd44(31)=dotproduct(ninjaE3,spvak2l4)
      acd44(32)=abb44(38)
      acd44(33)=dotproduct(ninjaE3,spvae1l5)
      acd44(34)=abb44(40)
      acd44(35)=dotproduct(ninjaE3,spval5k1)
      acd44(36)=abb44(41)
      acd44(37)=dotproduct(ninjaE3,spval4k1)
      acd44(38)=abb44(43)
      acd44(39)=dotproduct(ninjaE3,spvak1e1)
      acd44(40)=abb44(45)
      acd44(41)=dotproduct(ninjaE3,spvae2k1)
      acd44(42)=abb44(47)
      acd44(43)=dotproduct(ninjaE3,spval4e2)
      acd44(44)=abb44(49)
      acd44(45)=dotproduct(ninjaE3,spvak2e2)
      acd44(46)=abb44(50)
      acd44(47)=dotproduct(ninjaE3,spvae1k2)
      acd44(48)=abb44(52)
      acd44(49)=dotproduct(ninjaE3,spvak1e2)
      acd44(50)=abb44(54)
      acd44(51)=dotproduct(ninjaE3,spval5l4)
      acd44(52)=abb44(59)
      acd44(53)=dotproduct(ninjaE3,spvak2e1)
      acd44(54)=abb44(61)
      acd44(55)=dotproduct(ninjaE3,spval4l5)
      acd44(56)=abb44(63)
      acd44(57)=dotproduct(ninjaE3,spval4k2)
      acd44(58)=abb44(73)
      acd44(59)=acd44(2)*acd44(1)
      acd44(60)=acd44(4)*acd44(3)
      acd44(61)=acd44(6)*acd44(5)
      acd44(62)=acd44(8)*acd44(7)
      acd44(63)=acd44(10)*acd44(9)
      acd44(64)=acd44(12)*acd44(11)
      acd44(65)=acd44(14)*acd44(13)
      acd44(66)=acd44(16)*acd44(15)
      acd44(67)=acd44(18)*acd44(17)
      acd44(68)=acd44(20)*acd44(19)
      acd44(69)=acd44(22)*acd44(21)
      acd44(70)=acd44(24)*acd44(23)
      acd44(71)=acd44(26)*acd44(25)
      acd44(72)=acd44(28)*acd44(27)
      acd44(73)=acd44(30)*acd44(29)
      acd44(74)=acd44(32)*acd44(31)
      acd44(75)=acd44(34)*acd44(33)
      acd44(76)=acd44(36)*acd44(35)
      acd44(77)=acd44(38)*acd44(37)
      acd44(78)=acd44(40)*acd44(39)
      acd44(79)=acd44(42)*acd44(41)
      acd44(80)=acd44(44)*acd44(43)
      acd44(81)=acd44(46)*acd44(45)
      acd44(82)=acd44(48)*acd44(47)
      acd44(83)=acd44(50)*acd44(49)
      acd44(84)=acd44(52)*acd44(51)
      acd44(85)=acd44(54)*acd44(53)
      acd44(86)=acd44(56)*acd44(55)
      acd44(87)=acd44(58)*acd44(57)
      acd44(59)=acd44(87)+acd44(86)+acd44(85)+acd44(84)+acd44(83)+acd44(82)+acd&
      &44(81)+acd44(80)+acd44(79)+acd44(78)+acd44(77)+acd44(76)+acd44(75)+acd44&
      &(74)+acd44(73)+acd44(72)+acd44(71)+acd44(70)+acd44(69)+acd44(68)+acd44(6&
      &7)+acd44(66)+acd44(65)+acd44(64)+acd44(63)+acd44(62)+acd44(61)+acd44(59)&
      &+acd44(60)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd44(59)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd44h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(118) :: acd44
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd44(1)=dotproduct(k1,ninjaA1)
      acd44(2)=abb44(16)
      acd44(3)=dotproduct(k2,ninjaA1)
      acd44(4)=abb44(18)
      acd44(5)=dotproduct(l4,ninjaA1)
      acd44(6)=abb44(20)
      acd44(7)=dotproduct(ninjaA1,spvak1k2)
      acd44(8)=abb44(15)
      acd44(9)=dotproduct(ninjaA1,spvae2l4)
      acd44(10)=abb44(17)
      acd44(11)=dotproduct(ninjaA1,spvak1l4)
      acd44(12)=abb44(21)
      acd44(13)=dotproduct(ninjaA1,spvak2l5)
      acd44(14)=abb44(25)
      acd44(15)=dotproduct(ninjaA1,spvak2k1)
      acd44(16)=abb44(26)
      acd44(17)=dotproduct(ninjaA1,spvae2e1)
      acd44(18)=abb44(27)
      acd44(19)=dotproduct(ninjaA1,spvak1l5)
      acd44(20)=abb44(28)
      acd44(21)=dotproduct(ninjaA1,spval5e1)
      acd44(22)=abb44(31)
      acd44(23)=dotproduct(ninjaA1,spval4e1)
      acd44(24)=abb44(32)
      acd44(25)=dotproduct(ninjaA1,spvae1e2)
      acd44(26)=abb44(33)
      acd44(27)=dotproduct(ninjaA1,spvae1k1)
      acd44(28)=abb44(34)
      acd44(29)=dotproduct(ninjaA1,spvae1l4)
      acd44(30)=abb44(35)
      acd44(31)=dotproduct(ninjaA1,spvak2l4)
      acd44(32)=abb44(38)
      acd44(33)=dotproduct(ninjaA1,spvae1l5)
      acd44(34)=abb44(40)
      acd44(35)=dotproduct(ninjaA1,spval5k1)
      acd44(36)=abb44(41)
      acd44(37)=dotproduct(ninjaA1,spval4k1)
      acd44(38)=abb44(43)
      acd44(39)=dotproduct(ninjaA1,spvak1e1)
      acd44(40)=abb44(45)
      acd44(41)=dotproduct(ninjaA1,spvae2k1)
      acd44(42)=abb44(47)
      acd44(43)=dotproduct(ninjaA1,spval4e2)
      acd44(44)=abb44(49)
      acd44(45)=dotproduct(ninjaA1,spvak2e2)
      acd44(46)=abb44(50)
      acd44(47)=dotproduct(ninjaA1,spvae1k2)
      acd44(48)=abb44(52)
      acd44(49)=dotproduct(ninjaA1,spvak1e2)
      acd44(50)=abb44(54)
      acd44(51)=dotproduct(ninjaA1,spval5l4)
      acd44(52)=abb44(59)
      acd44(53)=dotproduct(ninjaA1,spvak2e1)
      acd44(54)=abb44(61)
      acd44(55)=dotproduct(ninjaA1,spval4l5)
      acd44(56)=abb44(63)
      acd44(57)=dotproduct(ninjaA1,spval4k2)
      acd44(58)=abb44(73)
      acd44(59)=dotproduct(k1,ninjaA0)
      acd44(60)=dotproduct(k2,ninjaA0)
      acd44(61)=dotproduct(l4,ninjaA0)
      acd44(62)=dotproduct(ninjaA0,spvak1k2)
      acd44(63)=dotproduct(ninjaA0,spvae2l4)
      acd44(64)=dotproduct(ninjaA0,spvak1l4)
      acd44(65)=dotproduct(ninjaA0,spvak2l5)
      acd44(66)=dotproduct(ninjaA0,spvak2k1)
      acd44(67)=dotproduct(ninjaA0,spvae2e1)
      acd44(68)=dotproduct(ninjaA0,spvak1l5)
      acd44(69)=dotproduct(ninjaA0,spval5e1)
      acd44(70)=dotproduct(ninjaA0,spval4e1)
      acd44(71)=dotproduct(ninjaA0,spvae1e2)
      acd44(72)=dotproduct(ninjaA0,spvae1k1)
      acd44(73)=dotproduct(ninjaA0,spvae1l4)
      acd44(74)=dotproduct(ninjaA0,spvak2l4)
      acd44(75)=dotproduct(ninjaA0,spvae1l5)
      acd44(76)=dotproduct(ninjaA0,spval5k1)
      acd44(77)=dotproduct(ninjaA0,spval4k1)
      acd44(78)=dotproduct(ninjaA0,spvak1e1)
      acd44(79)=dotproduct(ninjaA0,spvae2k1)
      acd44(80)=dotproduct(ninjaA0,spval4e2)
      acd44(81)=dotproduct(ninjaA0,spvak2e2)
      acd44(82)=dotproduct(ninjaA0,spvae1k2)
      acd44(83)=dotproduct(ninjaA0,spvak1e2)
      acd44(84)=dotproduct(ninjaA0,spval5l4)
      acd44(85)=dotproduct(ninjaA0,spvak2e1)
      acd44(86)=dotproduct(ninjaA0,spval4l5)
      acd44(87)=dotproduct(ninjaA0,spval4k2)
      acd44(88)=abb44(14)
      acd44(89)=acd44(1)*acd44(2)
      acd44(90)=acd44(3)*acd44(4)
      acd44(91)=acd44(5)*acd44(6)
      acd44(92)=acd44(7)*acd44(8)
      acd44(93)=acd44(9)*acd44(10)
      acd44(94)=acd44(11)*acd44(12)
      acd44(95)=acd44(13)*acd44(14)
      acd44(96)=acd44(15)*acd44(16)
      acd44(97)=acd44(17)*acd44(18)
      acd44(98)=acd44(19)*acd44(20)
      acd44(99)=acd44(21)*acd44(22)
      acd44(100)=acd44(23)*acd44(24)
      acd44(101)=acd44(25)*acd44(26)
      acd44(102)=acd44(27)*acd44(28)
      acd44(103)=acd44(29)*acd44(30)
      acd44(104)=acd44(31)*acd44(32)
      acd44(105)=acd44(33)*acd44(34)
      acd44(106)=acd44(35)*acd44(36)
      acd44(107)=acd44(37)*acd44(38)
      acd44(108)=acd44(39)*acd44(40)
      acd44(109)=acd44(41)*acd44(42)
      acd44(110)=acd44(43)*acd44(44)
      acd44(111)=acd44(45)*acd44(46)
      acd44(112)=acd44(47)*acd44(48)
      acd44(113)=acd44(49)*acd44(50)
      acd44(114)=acd44(51)*acd44(52)
      acd44(115)=acd44(53)*acd44(54)
      acd44(116)=acd44(55)*acd44(56)
      acd44(117)=acd44(57)*acd44(58)
      acd44(89)=acd44(117)+acd44(116)+acd44(115)+acd44(114)+acd44(113)+acd44(11&
      &2)+acd44(111)+acd44(110)+acd44(109)+acd44(108)+acd44(107)+acd44(106)+acd&
      &44(105)+acd44(104)+acd44(103)+acd44(102)+acd44(101)+acd44(100)+acd44(99)&
      &+acd44(98)+acd44(97)+acd44(96)+acd44(95)+acd44(94)+acd44(93)+acd44(92)+a&
      &cd44(91)+acd44(89)+acd44(90)
      acd44(90)=acd44(59)*acd44(2)
      acd44(91)=acd44(60)*acd44(4)
      acd44(92)=acd44(61)*acd44(6)
      acd44(93)=acd44(62)*acd44(8)
      acd44(94)=acd44(63)*acd44(10)
      acd44(95)=acd44(64)*acd44(12)
      acd44(96)=acd44(65)*acd44(14)
      acd44(97)=acd44(66)*acd44(16)
      acd44(98)=acd44(67)*acd44(18)
      acd44(99)=acd44(68)*acd44(20)
      acd44(100)=acd44(69)*acd44(22)
      acd44(101)=acd44(70)*acd44(24)
      acd44(102)=acd44(71)*acd44(26)
      acd44(103)=acd44(72)*acd44(28)
      acd44(104)=acd44(73)*acd44(30)
      acd44(105)=acd44(74)*acd44(32)
      acd44(106)=acd44(75)*acd44(34)
      acd44(107)=acd44(76)*acd44(36)
      acd44(108)=acd44(77)*acd44(38)
      acd44(109)=acd44(78)*acd44(40)
      acd44(110)=acd44(79)*acd44(42)
      acd44(111)=acd44(80)*acd44(44)
      acd44(112)=acd44(81)*acd44(46)
      acd44(113)=acd44(82)*acd44(48)
      acd44(114)=acd44(83)*acd44(50)
      acd44(115)=acd44(84)*acd44(52)
      acd44(116)=acd44(85)*acd44(54)
      acd44(117)=acd44(86)*acd44(56)
      acd44(118)=acd44(87)*acd44(58)
      acd44(90)=acd44(88)+acd44(118)+acd44(117)+acd44(116)+acd44(115)+acd44(114&
      &)+acd44(113)+acd44(112)+acd44(111)+acd44(110)+acd44(109)+acd44(108)+acd4&
      &4(107)+acd44(106)+acd44(105)+acd44(104)+acd44(103)+acd44(102)+acd44(101)&
      &+acd44(100)+acd44(99)+acd44(98)+acd44(97)+acd44(96)+acd44(95)+acd44(94)+&
      &acd44(93)+acd44(92)+acd44(90)+acd44(91)
      brack(ninjaidxt0x0mu0)=acd44(90)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd44(89)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d44h0_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd44h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA0(1:4) = - a0(0:3)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d44h0l132_qp
