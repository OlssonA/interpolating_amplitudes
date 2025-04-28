module     p2_gg_httbar_d38h4l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d38h4l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd38h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(69) :: acd38
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd38(1)=dotproduct(k2,ninjaE3)
      acd38(2)=abb38(15)
      acd38(3)=dotproduct(l5,ninjaE3)
      acd38(4)=abb38(21)
      acd38(5)=dotproduct(ninjaE3,spval5k2)
      acd38(6)=abb38(14)
      acd38(7)=dotproduct(ninjaE3,spvak2l5)
      acd38(8)=abb38(16)
      acd38(9)=dotproduct(ninjaE3,spvae2k2)
      acd38(10)=abb38(17)
      acd38(11)=dotproduct(ninjaE3,spvak2e1)
      acd38(12)=abb38(18)
      acd38(13)=dotproduct(ninjaE3,spvae2k1)
      acd38(14)=abb38(20)
      acd38(15)=dotproduct(ninjaE3,spvae2e1)
      acd38(16)=abb38(22)
      acd38(17)=dotproduct(ninjaE3,spvae1e2)
      acd38(18)=abb38(23)
      acd38(19)=dotproduct(ninjaE3,spvak1e2)
      acd38(20)=abb38(24)
      acd38(21)=dotproduct(ninjaE3,spvak2e2)
      acd38(22)=abb38(25)
      acd38(23)=dotproduct(ninjaE3,spval5k1)
      acd38(24)=abb38(26)
      acd38(25)=dotproduct(ninjaE3,spval5l4)
      acd38(26)=abb38(27)
      acd38(27)=dotproduct(ninjaE3,spvak2k1)
      acd38(28)=abb38(28)
      acd38(29)=dotproduct(ninjaE3,spvak2l4)
      acd38(30)=abb38(29)
      acd38(31)=dotproduct(ninjaE3,spvak1l5)
      acd38(32)=abb38(30)
      acd38(33)=dotproduct(ninjaE3,spval4l5)
      acd38(34)=abb38(32)
      acd38(35)=dotproduct(ninjaE3,spvae2l5)
      acd38(36)=abb38(34)
      acd38(37)=dotproduct(ninjaE3,spvae1l5)
      acd38(38)=abb38(44)
      acd38(39)=dotproduct(ninjaE3,spval5e1)
      acd38(40)=abb38(50)
      acd38(41)=dotproduct(ninjaE3,spval5e2)
      acd38(42)=abb38(51)
      acd38(43)=dotproduct(ninjaE3,spvae2l4)
      acd38(44)=abb38(53)
      acd38(45)=dotproduct(ninjaE3,spval4e2)
      acd38(46)=abb38(58)
      acd38(47)=acd38(2)*acd38(1)
      acd38(48)=acd38(4)*acd38(3)
      acd38(49)=acd38(6)*acd38(5)
      acd38(50)=acd38(8)*acd38(7)
      acd38(51)=acd38(10)*acd38(9)
      acd38(52)=acd38(12)*acd38(11)
      acd38(53)=acd38(14)*acd38(13)
      acd38(54)=acd38(16)*acd38(15)
      acd38(55)=acd38(18)*acd38(17)
      acd38(56)=acd38(20)*acd38(19)
      acd38(57)=acd38(22)*acd38(21)
      acd38(58)=acd38(24)*acd38(23)
      acd38(59)=acd38(26)*acd38(25)
      acd38(60)=acd38(28)*acd38(27)
      acd38(61)=acd38(30)*acd38(29)
      acd38(62)=acd38(32)*acd38(31)
      acd38(63)=acd38(34)*acd38(33)
      acd38(64)=acd38(36)*acd38(35)
      acd38(65)=acd38(38)*acd38(37)
      acd38(66)=acd38(40)*acd38(39)
      acd38(67)=acd38(42)*acd38(41)
      acd38(68)=acd38(44)*acd38(43)
      acd38(69)=acd38(46)*acd38(45)
      acd38(47)=acd38(69)+acd38(68)+acd38(67)+acd38(66)+acd38(65)+acd38(64)+acd&
      &38(63)+acd38(62)+acd38(61)+acd38(60)+acd38(59)+acd38(58)+acd38(57)+acd38&
      &(56)+acd38(55)+acd38(54)+acd38(53)+acd38(52)+acd38(51)+acd38(50)+acd38(4&
      &9)+acd38(47)+acd38(48)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd38(47)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd38h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(94) :: acd38
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd38(1)=dotproduct(k2,ninjaA1)
      acd38(2)=abb38(15)
      acd38(3)=dotproduct(l5,ninjaA1)
      acd38(4)=abb38(21)
      acd38(5)=dotproduct(ninjaA1,spval5k2)
      acd38(6)=abb38(14)
      acd38(7)=dotproduct(ninjaA1,spvak2l5)
      acd38(8)=abb38(16)
      acd38(9)=dotproduct(ninjaA1,spvae2k2)
      acd38(10)=abb38(17)
      acd38(11)=dotproduct(ninjaA1,spvak2e1)
      acd38(12)=abb38(18)
      acd38(13)=dotproduct(ninjaA1,spvae2k1)
      acd38(14)=abb38(20)
      acd38(15)=dotproduct(ninjaA1,spvae2e1)
      acd38(16)=abb38(22)
      acd38(17)=dotproduct(ninjaA1,spvae1e2)
      acd38(18)=abb38(23)
      acd38(19)=dotproduct(ninjaA1,spvak1e2)
      acd38(20)=abb38(24)
      acd38(21)=dotproduct(ninjaA1,spvak2e2)
      acd38(22)=abb38(25)
      acd38(23)=dotproduct(ninjaA1,spval5k1)
      acd38(24)=abb38(26)
      acd38(25)=dotproduct(ninjaA1,spval5l4)
      acd38(26)=abb38(27)
      acd38(27)=dotproduct(ninjaA1,spvak2k1)
      acd38(28)=abb38(28)
      acd38(29)=dotproduct(ninjaA1,spvak2l4)
      acd38(30)=abb38(29)
      acd38(31)=dotproduct(ninjaA1,spvak1l5)
      acd38(32)=abb38(30)
      acd38(33)=dotproduct(ninjaA1,spval4l5)
      acd38(34)=abb38(32)
      acd38(35)=dotproduct(ninjaA1,spvae2l5)
      acd38(36)=abb38(34)
      acd38(37)=dotproduct(ninjaA1,spvae1l5)
      acd38(38)=abb38(44)
      acd38(39)=dotproduct(ninjaA1,spval5e1)
      acd38(40)=abb38(50)
      acd38(41)=dotproduct(ninjaA1,spval5e2)
      acd38(42)=abb38(51)
      acd38(43)=dotproduct(ninjaA1,spvae2l4)
      acd38(44)=abb38(53)
      acd38(45)=dotproduct(ninjaA1,spval4e2)
      acd38(46)=abb38(58)
      acd38(47)=dotproduct(k2,ninjaA0)
      acd38(48)=dotproduct(l5,ninjaA0)
      acd38(49)=dotproduct(ninjaA0,spval5k2)
      acd38(50)=dotproduct(ninjaA0,spvak2l5)
      acd38(51)=dotproduct(ninjaA0,spvae2k2)
      acd38(52)=dotproduct(ninjaA0,spvak2e1)
      acd38(53)=dotproduct(ninjaA0,spvae2k1)
      acd38(54)=dotproduct(ninjaA0,spvae2e1)
      acd38(55)=dotproduct(ninjaA0,spvae1e2)
      acd38(56)=dotproduct(ninjaA0,spvak1e2)
      acd38(57)=dotproduct(ninjaA0,spvak2e2)
      acd38(58)=dotproduct(ninjaA0,spval5k1)
      acd38(59)=dotproduct(ninjaA0,spval5l4)
      acd38(60)=dotproduct(ninjaA0,spvak2k1)
      acd38(61)=dotproduct(ninjaA0,spvak2l4)
      acd38(62)=dotproduct(ninjaA0,spvak1l5)
      acd38(63)=dotproduct(ninjaA0,spval4l5)
      acd38(64)=dotproduct(ninjaA0,spvae2l5)
      acd38(65)=dotproduct(ninjaA0,spvae1l5)
      acd38(66)=dotproduct(ninjaA0,spval5e1)
      acd38(67)=dotproduct(ninjaA0,spval5e2)
      acd38(68)=dotproduct(ninjaA0,spvae2l4)
      acd38(69)=dotproduct(ninjaA0,spval4e2)
      acd38(70)=abb38(19)
      acd38(71)=acd38(1)*acd38(2)
      acd38(72)=acd38(3)*acd38(4)
      acd38(73)=acd38(5)*acd38(6)
      acd38(74)=acd38(7)*acd38(8)
      acd38(75)=acd38(9)*acd38(10)
      acd38(76)=acd38(11)*acd38(12)
      acd38(77)=acd38(13)*acd38(14)
      acd38(78)=acd38(15)*acd38(16)
      acd38(79)=acd38(17)*acd38(18)
      acd38(80)=acd38(19)*acd38(20)
      acd38(81)=acd38(21)*acd38(22)
      acd38(82)=acd38(23)*acd38(24)
      acd38(83)=acd38(25)*acd38(26)
      acd38(84)=acd38(27)*acd38(28)
      acd38(85)=acd38(29)*acd38(30)
      acd38(86)=acd38(31)*acd38(32)
      acd38(87)=acd38(33)*acd38(34)
      acd38(88)=acd38(35)*acd38(36)
      acd38(89)=acd38(37)*acd38(38)
      acd38(90)=acd38(39)*acd38(40)
      acd38(91)=acd38(41)*acd38(42)
      acd38(92)=acd38(43)*acd38(44)
      acd38(93)=acd38(45)*acd38(46)
      acd38(71)=acd38(93)+acd38(92)+acd38(91)+acd38(90)+acd38(89)+acd38(88)+acd&
      &38(87)+acd38(86)+acd38(85)+acd38(84)+acd38(83)+acd38(82)+acd38(81)+acd38&
      &(80)+acd38(79)+acd38(78)+acd38(77)+acd38(76)+acd38(75)+acd38(74)+acd38(7&
      &3)+acd38(71)+acd38(72)
      acd38(72)=acd38(47)*acd38(2)
      acd38(73)=acd38(48)*acd38(4)
      acd38(74)=acd38(49)*acd38(6)
      acd38(75)=acd38(50)*acd38(8)
      acd38(76)=acd38(51)*acd38(10)
      acd38(77)=acd38(52)*acd38(12)
      acd38(78)=acd38(53)*acd38(14)
      acd38(79)=acd38(54)*acd38(16)
      acd38(80)=acd38(55)*acd38(18)
      acd38(81)=acd38(56)*acd38(20)
      acd38(82)=acd38(57)*acd38(22)
      acd38(83)=acd38(58)*acd38(24)
      acd38(84)=acd38(59)*acd38(26)
      acd38(85)=acd38(60)*acd38(28)
      acd38(86)=acd38(61)*acd38(30)
      acd38(87)=acd38(62)*acd38(32)
      acd38(88)=acd38(63)*acd38(34)
      acd38(89)=acd38(64)*acd38(36)
      acd38(90)=acd38(65)*acd38(38)
      acd38(91)=acd38(66)*acd38(40)
      acd38(92)=acd38(67)*acd38(42)
      acd38(93)=acd38(68)*acd38(44)
      acd38(94)=acd38(69)*acd38(46)
      acd38(72)=acd38(70)+acd38(94)+acd38(93)+acd38(92)+acd38(91)+acd38(90)+acd&
      &38(89)+acd38(88)+acd38(87)+acd38(86)+acd38(85)+acd38(84)+acd38(83)+acd38&
      &(82)+acd38(81)+acd38(80)+acd38(79)+acd38(78)+acd38(77)+acd38(76)+acd38(7&
      &5)+acd38(74)+acd38(72)+acd38(73)
      brack(ninjaidxt0x0mu0)=acd38(72)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd38(71)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d38h4_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd38h4
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
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d38h4l132
