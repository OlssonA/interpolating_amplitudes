module     p2_gg_httbar_d148h12l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d148h12l132.f90
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
      use p2_gg_httbar_abbrevd148h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(28) :: acd148
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd148(1)=dotproduct(k2,ninjaE3)
      acd148(2)=dotproduct(e2,ninjaE3)
      acd148(3)=abb148(65)
      acd148(4)=dotproduct(l4,ninjaE3)
      acd148(5)=abb148(57)
      acd148(6)=dotproduct(ninjaE3,spvak1l4)
      acd148(7)=abb148(12)
      acd148(8)=dotproduct(ninjaE3,spvak2l4)
      acd148(9)=abb148(15)
      acd148(10)=dotproduct(ninjaE3,spvak2k1)
      acd148(11)=abb148(17)
      acd148(12)=dotproduct(ninjaE3,spval5l4)
      acd148(13)=abb148(30)
      acd148(14)=dotproduct(ninjaE3,spvak2l5)
      acd148(15)=abb148(31)
      acd148(16)=dotproduct(ninjaE3,spvak2e1)
      acd148(17)=abb148(45)
      acd148(18)=dotproduct(ninjaE3,spvae1l4)
      acd148(19)=abb148(91)
      acd148(20)=acd148(3)*acd148(1)
      acd148(21)=acd148(5)*acd148(4)
      acd148(22)=acd148(7)*acd148(6)
      acd148(23)=acd148(9)*acd148(8)
      acd148(24)=acd148(11)*acd148(10)
      acd148(25)=acd148(13)*acd148(12)
      acd148(26)=acd148(15)*acd148(14)
      acd148(27)=acd148(17)*acd148(16)
      acd148(28)=acd148(19)*acd148(18)
      acd148(20)=acd148(28)+acd148(27)+acd148(26)+acd148(25)+acd148(24)+acd148(&
      &23)+acd148(22)+acd148(20)+acd148(21)
      acd148(20)=acd148(2)*acd148(20)
      brack(ninjaidxt1x0mu0)=acd148(20)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd148h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(109) :: acd148
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd148(1)=dotproduct(k2,ninjaA1)
      acd148(2)=dotproduct(e2,ninjaE3)
      acd148(3)=abb148(65)
      acd148(4)=dotproduct(k2,ninjaE3)
      acd148(5)=dotproduct(e2,ninjaA1)
      acd148(6)=dotproduct(l4,ninjaA1)
      acd148(7)=abb148(57)
      acd148(8)=dotproduct(l4,ninjaE3)
      acd148(9)=dotproduct(ninjaE3,spvak2l4)
      acd148(10)=abb148(15)
      acd148(11)=dotproduct(ninjaE3,spvak1l4)
      acd148(12)=abb148(12)
      acd148(13)=dotproduct(ninjaE3,spvak2k1)
      acd148(14)=abb148(17)
      acd148(15)=dotproduct(ninjaE3,spval5l4)
      acd148(16)=abb148(30)
      acd148(17)=dotproduct(ninjaE3,spvak2l5)
      acd148(18)=abb148(31)
      acd148(19)=dotproduct(ninjaE3,spvae1l4)
      acd148(20)=abb148(91)
      acd148(21)=dotproduct(ninjaE3,spvak2e1)
      acd148(22)=abb148(45)
      acd148(23)=dotproduct(ninjaA1,spvak2l4)
      acd148(24)=dotproduct(ninjaA1,spvak1l4)
      acd148(25)=dotproduct(ninjaA1,spvak2k1)
      acd148(26)=dotproduct(ninjaA1,spval5l4)
      acd148(27)=dotproduct(ninjaA1,spvak2l5)
      acd148(28)=dotproduct(ninjaA1,spvae1l4)
      acd148(29)=dotproduct(ninjaA1,spvak2e1)
      acd148(30)=dotproduct(k2,ninjaA0)
      acd148(31)=dotproduct(e2,ninjaA0)
      acd148(32)=abb148(13)
      acd148(33)=dotproduct(l4,ninjaA0)
      acd148(34)=abb148(38)
      acd148(35)=dotproduct(ninjaA0,spvak2l4)
      acd148(36)=dotproduct(ninjaA0,spvak1l4)
      acd148(37)=dotproduct(ninjaA0,spvak2k1)
      acd148(38)=dotproduct(ninjaA0,spval5l4)
      acd148(39)=dotproduct(ninjaA0,spvak2l5)
      acd148(40)=dotproduct(ninjaA0,spvae1l4)
      acd148(41)=dotproduct(ninjaA0,spvak2e1)
      acd148(42)=abb148(43)
      acd148(43)=dotproduct(ninjaA0,ninjaE3)
      acd148(44)=abb148(105)
      acd148(45)=abb148(11)
      acd148(46)=abb148(18)
      acd148(47)=dotproduct(ninjaE3,spvak2e2)
      acd148(48)=abb148(14)
      acd148(49)=dotproduct(ninjaE3,spvae2k2)
      acd148(50)=abb148(16)
      acd148(51)=dotproduct(ninjaE3,spval4k2)
      acd148(52)=abb148(19)
      acd148(53)=dotproduct(ninjaE3,spvak1k2)
      acd148(54)=abb148(20)
      acd148(55)=dotproduct(ninjaE3,spvae1k2)
      acd148(56)=abb148(24)
      acd148(57)=dotproduct(ninjaE3,spval4e1)
      acd148(58)=abb148(26)
      acd148(59)=dotproduct(ninjaE3,spval4k1)
      acd148(60)=abb148(27)
      acd148(61)=abb148(55)
      acd148(62)=dotproduct(ninjaE3,spvak1e2)
      acd148(63)=abb148(35)
      acd148(64)=abb148(41)
      acd148(65)=dotproduct(ninjaE3,spvae2l4)
      acd148(66)=abb148(46)
      acd148(67)=dotproduct(ninjaE3,spvae2e1)
      acd148(68)=abb148(47)
      acd148(69)=dotproduct(ninjaE3,spvae2k1)
      acd148(70)=abb148(50)
      acd148(71)=dotproduct(ninjaE3,spval5k2)
      acd148(72)=abb148(58)
      acd148(73)=dotproduct(ninjaE3,spval4l5)
      acd148(74)=abb148(59)
      acd148(75)=dotproduct(ninjaE3,spval4e2)
      acd148(76)=abb148(66)
      acd148(77)=dotproduct(ninjaE3,spvae1e2)
      acd148(78)=abb148(76)
      acd148(79)=dotproduct(ninjaE3,spvae2l5)
      acd148(80)=abb148(94)
      acd148(81)=dotproduct(ninjaE3,spval5e2)
      acd148(82)=abb148(145)
      acd148(83)=acd148(22)*acd148(21)
      acd148(84)=acd148(20)*acd148(19)
      acd148(85)=acd148(18)*acd148(17)
      acd148(86)=acd148(16)*acd148(15)
      acd148(87)=acd148(14)*acd148(13)
      acd148(88)=acd148(12)*acd148(11)
      acd148(89)=acd148(10)*acd148(9)
      acd148(90)=acd148(7)*acd148(8)
      acd148(91)=acd148(3)*acd148(4)
      acd148(83)=acd148(91)+acd148(87)+acd148(88)+acd148(89)+acd148(90)+acd148(&
      &83)+acd148(84)+acd148(85)+acd148(86)
      acd148(84)=acd148(5)*acd148(83)
      acd148(85)=acd148(22)*acd148(29)
      acd148(86)=acd148(20)*acd148(28)
      acd148(87)=acd148(18)*acd148(27)
      acd148(88)=acd148(16)*acd148(26)
      acd148(89)=acd148(14)*acd148(25)
      acd148(90)=acd148(12)*acd148(24)
      acd148(91)=acd148(10)*acd148(23)
      acd148(92)=acd148(7)*acd148(6)
      acd148(93)=acd148(3)*acd148(1)
      acd148(85)=acd148(93)+acd148(92)+acd148(91)+acd148(90)+acd148(89)+acd148(&
      &88)+acd148(87)+acd148(85)+acd148(86)
      acd148(85)=acd148(2)*acd148(85)
      acd148(84)=acd148(84)+acd148(85)
      acd148(83)=acd148(31)*acd148(83)
      acd148(85)=acd148(22)*acd148(41)
      acd148(86)=acd148(20)*acd148(40)
      acd148(87)=acd148(18)*acd148(39)
      acd148(88)=acd148(16)*acd148(38)
      acd148(89)=acd148(14)*acd148(37)
      acd148(90)=acd148(12)*acd148(36)
      acd148(91)=acd148(10)*acd148(35)
      acd148(92)=acd148(7)*acd148(33)
      acd148(93)=acd148(3)*acd148(30)
      acd148(85)=acd148(93)+acd148(92)+acd148(91)+acd148(90)+acd148(89)+acd148(&
      &88)+acd148(87)+acd148(86)+acd148(42)+acd148(85)
      acd148(85)=acd148(2)*acd148(85)
      acd148(86)=acd148(81)*acd148(82)
      acd148(87)=acd148(79)*acd148(80)
      acd148(88)=acd148(77)*acd148(78)
      acd148(89)=acd148(75)*acd148(76)
      acd148(90)=acd148(73)*acd148(74)
      acd148(91)=acd148(71)*acd148(72)
      acd148(92)=acd148(69)*acd148(70)
      acd148(93)=acd148(67)*acd148(68)
      acd148(94)=acd148(65)*acd148(66)
      acd148(95)=acd148(62)*acd148(63)
      acd148(96)=acd148(59)*acd148(60)
      acd148(97)=acd148(57)*acd148(58)
      acd148(98)=acd148(55)*acd148(56)
      acd148(99)=acd148(53)*acd148(54)
      acd148(100)=acd148(51)*acd148(52)
      acd148(101)=acd148(49)*acd148(50)
      acd148(102)=acd148(47)*acd148(48)
      acd148(103)=acd148(43)*acd148(44)
      acd148(104)=acd148(19)*acd148(64)
      acd148(105)=acd148(15)*acd148(61)
      acd148(106)=acd148(11)*acd148(46)
      acd148(107)=acd148(9)*acd148(45)
      acd148(108)=acd148(8)*acd148(34)
      acd148(109)=acd148(4)*acd148(32)
      acd148(83)=acd148(85)+acd148(83)+acd148(109)+acd148(108)+acd148(107)+acd1&
      &48(106)+acd148(105)+acd148(104)-2.0_ki*acd148(103)+acd148(102)+acd148(10&
      &1)+acd148(100)+acd148(99)+acd148(98)+acd148(97)+acd148(96)+acd148(95)+ac&
      &d148(94)+acd148(93)+acd148(92)+acd148(91)+acd148(90)+acd148(89)+acd148(8&
      &8)+acd148(86)+acd148(87)
      brack(ninjaidxt0x0mu0)=acd148(83)
      brack(ninjaidxt0x1mu0)=acd148(84)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d148h12_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd148h12
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
end module     p2_gg_httbar_d148h12l132
