module     p2_gg_httbar_d133h0l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d133h0l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd133h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(28) :: acd133
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd133(1)=dotproduct(k2,ninjaE3)
      acd133(2)=dotproduct(e2,ninjaE3)
      acd133(3)=abb133(65)
      acd133(4)=dotproduct(l5,ninjaE3)
      acd133(5)=abb133(57)
      acd133(6)=dotproduct(ninjaE3,spval5k1)
      acd133(7)=abb133(12)
      acd133(8)=dotproduct(ninjaE3,spval5k2)
      acd133(9)=abb133(15)
      acd133(10)=dotproduct(ninjaE3,spvak1k2)
      acd133(11)=abb133(22)
      acd133(12)=dotproduct(ninjaE3,spvae1k2)
      acd133(13)=abb133(24)
      acd133(14)=dotproduct(ninjaE3,spval4k2)
      acd133(15)=abb133(25)
      acd133(16)=dotproduct(ninjaE3,spval5l4)
      acd133(17)=abb133(30)
      acd133(18)=dotproduct(ninjaE3,spval5e1)
      acd133(19)=abb133(86)
      acd133(20)=acd133(3)*acd133(1)
      acd133(21)=acd133(5)*acd133(4)
      acd133(22)=acd133(7)*acd133(6)
      acd133(23)=acd133(9)*acd133(8)
      acd133(24)=acd133(11)*acd133(10)
      acd133(25)=acd133(13)*acd133(12)
      acd133(26)=acd133(15)*acd133(14)
      acd133(27)=acd133(17)*acd133(16)
      acd133(28)=acd133(19)*acd133(18)
      acd133(20)=acd133(28)+acd133(27)+acd133(26)+acd133(25)+acd133(24)+acd133(&
      &23)+acd133(22)+acd133(20)+acd133(21)
      acd133(20)=acd133(2)*acd133(20)
      brack(ninjaidxt1x0mu0)=acd133(20)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd133h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(109) :: acd133
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd133(1)=dotproduct(k2,ninjaA1)
      acd133(2)=dotproduct(e2,ninjaE3)
      acd133(3)=abb133(65)
      acd133(4)=dotproduct(k2,ninjaE3)
      acd133(5)=dotproduct(e2,ninjaA1)
      acd133(6)=dotproduct(l5,ninjaA1)
      acd133(7)=abb133(57)
      acd133(8)=dotproduct(l5,ninjaE3)
      acd133(9)=dotproduct(ninjaE3,spval5k1)
      acd133(10)=abb133(12)
      acd133(11)=dotproduct(ninjaE3,spval5k2)
      acd133(12)=abb133(15)
      acd133(13)=dotproduct(ninjaE3,spvak1k2)
      acd133(14)=abb133(22)
      acd133(15)=dotproduct(ninjaE3,spvae1k2)
      acd133(16)=abb133(24)
      acd133(17)=dotproduct(ninjaE3,spval4k2)
      acd133(18)=abb133(25)
      acd133(19)=dotproduct(ninjaE3,spval5l4)
      acd133(20)=abb133(30)
      acd133(21)=dotproduct(ninjaE3,spval5e1)
      acd133(22)=abb133(86)
      acd133(23)=dotproduct(ninjaA1,spval5k1)
      acd133(24)=dotproduct(ninjaA1,spval5k2)
      acd133(25)=dotproduct(ninjaA1,spvak1k2)
      acd133(26)=dotproduct(ninjaA1,spvae1k2)
      acd133(27)=dotproduct(ninjaA1,spval4k2)
      acd133(28)=dotproduct(ninjaA1,spval5l4)
      acd133(29)=dotproduct(ninjaA1,spval5e1)
      acd133(30)=dotproduct(k2,ninjaA0)
      acd133(31)=dotproduct(e2,ninjaA0)
      acd133(32)=abb133(13)
      acd133(33)=dotproduct(l5,ninjaA0)
      acd133(34)=abb133(38)
      acd133(35)=dotproduct(ninjaA0,spval5k1)
      acd133(36)=dotproduct(ninjaA0,spval5k2)
      acd133(37)=dotproduct(ninjaA0,spvak1k2)
      acd133(38)=dotproduct(ninjaA0,spvae1k2)
      acd133(39)=dotproduct(ninjaA0,spval4k2)
      acd133(40)=dotproduct(ninjaA0,spval5l4)
      acd133(41)=dotproduct(ninjaA0,spval5e1)
      acd133(42)=abb133(43)
      acd133(43)=dotproduct(ninjaA0,ninjaE3)
      acd133(44)=abb133(105)
      acd133(45)=dotproduct(ninjaE3,spvak2l5)
      acd133(46)=abb133(11)
      acd133(47)=abb133(17)
      acd133(48)=dotproduct(ninjaE3,spvae2k2)
      acd133(49)=abb133(14)
      acd133(50)=abb133(18)
      acd133(51)=dotproduct(ninjaE3,spvak2e2)
      acd133(52)=abb133(16)
      acd133(53)=dotproduct(ninjaE3,spvak2l4)
      acd133(54)=abb133(19)
      acd133(55)=dotproduct(ninjaE3,spvak1l5)
      acd133(56)=abb133(20)
      acd133(57)=dotproduct(ninjaE3,spval4l5)
      acd133(58)=abb133(21)
      acd133(59)=dotproduct(ninjaE3,spvak2k1)
      acd133(60)=abb133(23)
      acd133(61)=dotproduct(ninjaE3,spvae1l5)
      acd133(62)=abb133(26)
      acd133(63)=abb133(55)
      acd133(64)=dotproduct(ninjaE3,spvae2k1)
      acd133(65)=abb133(35)
      acd133(66)=dotproduct(ninjaE3,spvak2e1)
      acd133(67)=abb133(39)
      acd133(68)=abb133(41)
      acd133(69)=dotproduct(ninjaE3,spval5e2)
      acd133(70)=abb133(46)
      acd133(71)=dotproduct(ninjaE3,spvae1e2)
      acd133(72)=abb133(47)
      acd133(73)=dotproduct(ninjaE3,spvak1e2)
      acd133(74)=abb133(50)
      acd133(75)=dotproduct(ninjaE3,spvae2l5)
      acd133(76)=abb133(69)
      acd133(77)=dotproduct(ninjaE3,spvae2e1)
      acd133(78)=abb133(94)
      acd133(79)=dotproduct(ninjaE3,spvae2l4)
      acd133(80)=abb133(120)
      acd133(81)=dotproduct(ninjaE3,spval4e2)
      acd133(82)=abb133(125)
      acd133(83)=acd133(22)*acd133(21)
      acd133(84)=acd133(20)*acd133(19)
      acd133(85)=acd133(18)*acd133(17)
      acd133(86)=acd133(16)*acd133(15)
      acd133(87)=acd133(14)*acd133(13)
      acd133(88)=acd133(12)*acd133(11)
      acd133(89)=acd133(10)*acd133(9)
      acd133(90)=acd133(7)*acd133(8)
      acd133(91)=acd133(3)*acd133(4)
      acd133(83)=acd133(91)+acd133(87)+acd133(88)+acd133(89)+acd133(90)+acd133(&
      &83)+acd133(84)+acd133(85)+acd133(86)
      acd133(84)=acd133(5)*acd133(83)
      acd133(85)=acd133(22)*acd133(29)
      acd133(86)=acd133(20)*acd133(28)
      acd133(87)=acd133(18)*acd133(27)
      acd133(88)=acd133(16)*acd133(26)
      acd133(89)=acd133(14)*acd133(25)
      acd133(90)=acd133(12)*acd133(24)
      acd133(91)=acd133(10)*acd133(23)
      acd133(92)=acd133(7)*acd133(6)
      acd133(93)=acd133(3)*acd133(1)
      acd133(85)=acd133(93)+acd133(92)+acd133(91)+acd133(90)+acd133(89)+acd133(&
      &88)+acd133(87)+acd133(85)+acd133(86)
      acd133(85)=acd133(2)*acd133(85)
      acd133(84)=acd133(84)+acd133(85)
      acd133(83)=acd133(31)*acd133(83)
      acd133(85)=acd133(22)*acd133(41)
      acd133(86)=acd133(20)*acd133(40)
      acd133(87)=acd133(18)*acd133(39)
      acd133(88)=acd133(16)*acd133(38)
      acd133(89)=acd133(14)*acd133(37)
      acd133(90)=acd133(12)*acd133(36)
      acd133(91)=acd133(10)*acd133(35)
      acd133(92)=acd133(7)*acd133(33)
      acd133(93)=acd133(3)*acd133(30)
      acd133(85)=acd133(93)+acd133(92)+acd133(91)+acd133(90)+acd133(89)+acd133(&
      &88)+acd133(87)+acd133(86)+acd133(42)+acd133(85)
      acd133(85)=acd133(2)*acd133(85)
      acd133(86)=acd133(81)*acd133(82)
      acd133(87)=acd133(79)*acd133(80)
      acd133(88)=acd133(77)*acd133(78)
      acd133(89)=acd133(75)*acd133(76)
      acd133(90)=acd133(73)*acd133(74)
      acd133(91)=acd133(71)*acd133(72)
      acd133(92)=acd133(69)*acd133(70)
      acd133(93)=acd133(66)*acd133(67)
      acd133(94)=acd133(64)*acd133(65)
      acd133(95)=acd133(61)*acd133(62)
      acd133(96)=acd133(59)*acd133(60)
      acd133(97)=acd133(57)*acd133(58)
      acd133(98)=acd133(55)*acd133(56)
      acd133(99)=acd133(53)*acd133(54)
      acd133(100)=acd133(51)*acd133(52)
      acd133(101)=acd133(48)*acd133(49)
      acd133(102)=acd133(45)*acd133(46)
      acd133(103)=acd133(43)*acd133(44)
      acd133(104)=acd133(21)*acd133(68)
      acd133(105)=acd133(19)*acd133(63)
      acd133(106)=acd133(11)*acd133(50)
      acd133(107)=acd133(9)*acd133(47)
      acd133(108)=acd133(8)*acd133(34)
      acd133(109)=acd133(4)*acd133(32)
      acd133(83)=acd133(85)+acd133(83)+acd133(109)+acd133(108)+acd133(107)+acd1&
      &33(106)+acd133(105)+acd133(104)-2.0_ki*acd133(103)+acd133(102)+acd133(10&
      &1)+acd133(100)+acd133(99)+acd133(98)+acd133(97)+acd133(96)+acd133(95)+ac&
      &d133(94)+acd133(93)+acd133(92)+acd133(91)+acd133(90)+acd133(89)+acd133(8&
      &8)+acd133(86)+acd133(87)
      brack(ninjaidxt0x0mu0)=acd133(83)
      brack(ninjaidxt0x1mu0)=acd133(84)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d133h0_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd133h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA0(1:4) = + a0(0:3)
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
end module     p2_gg_httbar_d133h0l132
