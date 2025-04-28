module     p2_gg_httbar_d40h0l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d40h0l132.f90
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
      use p2_gg_httbar_abbrevd40h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(69) :: acd40
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd40(1)=dotproduct(k2,ninjaE3)
      acd40(2)=abb40(27)
      acd40(3)=dotproduct(l4,ninjaE3)
      acd40(4)=abb40(16)
      acd40(5)=dotproduct(ninjaE3,spvae2k2)
      acd40(6)=abb40(14)
      acd40(7)=dotproduct(ninjaE3,spvak2l4)
      acd40(8)=abb40(15)
      acd40(9)=dotproduct(ninjaE3,spvak2e2)
      acd40(10)=abb40(17)
      acd40(11)=dotproduct(ninjaE3,spvak2l5)
      acd40(12)=abb40(18)
      acd40(13)=dotproduct(ninjaE3,spvae2l4)
      acd40(14)=abb40(19)
      acd40(15)=dotproduct(ninjaE3,spvak2e1)
      acd40(16)=abb40(20)
      acd40(17)=dotproduct(ninjaE3,spvae2k1)
      acd40(18)=abb40(21)
      acd40(19)=dotproduct(ninjaE3,spvak2k1)
      acd40(20)=abb40(22)
      acd40(21)=dotproduct(ninjaE3,spvak1l4)
      acd40(22)=abb40(24)
      acd40(23)=dotproduct(ninjaE3,spvak1e2)
      acd40(24)=abb40(25)
      acd40(25)=dotproduct(ninjaE3,spval5l4)
      acd40(26)=abb40(29)
      acd40(27)=dotproduct(ninjaE3,spvae2e1)
      acd40(28)=abb40(30)
      acd40(29)=dotproduct(ninjaE3,spval4k2)
      acd40(30)=abb40(31)
      acd40(31)=dotproduct(ninjaE3,spvae1e2)
      acd40(32)=abb40(32)
      acd40(33)=dotproduct(ninjaE3,spval4e1)
      acd40(34)=abb40(33)
      acd40(35)=dotproduct(ninjaE3,spvae2l5)
      acd40(36)=abb40(36)
      acd40(37)=dotproduct(ninjaE3,spval4l5)
      acd40(38)=abb40(37)
      acd40(39)=dotproduct(ninjaE3,spval4k1)
      acd40(40)=abb40(39)
      acd40(41)=dotproduct(ninjaE3,spval5e2)
      acd40(42)=abb40(44)
      acd40(43)=dotproduct(ninjaE3,spvae1l4)
      acd40(44)=abb40(46)
      acd40(45)=dotproduct(ninjaE3,spval4e2)
      acd40(46)=abb40(50)
      acd40(47)=acd40(2)*acd40(1)
      acd40(48)=acd40(4)*acd40(3)
      acd40(49)=acd40(6)*acd40(5)
      acd40(50)=acd40(8)*acd40(7)
      acd40(51)=acd40(10)*acd40(9)
      acd40(52)=acd40(12)*acd40(11)
      acd40(53)=acd40(14)*acd40(13)
      acd40(54)=acd40(16)*acd40(15)
      acd40(55)=acd40(18)*acd40(17)
      acd40(56)=acd40(20)*acd40(19)
      acd40(57)=acd40(22)*acd40(21)
      acd40(58)=acd40(24)*acd40(23)
      acd40(59)=acd40(26)*acd40(25)
      acd40(60)=acd40(28)*acd40(27)
      acd40(61)=acd40(30)*acd40(29)
      acd40(62)=acd40(32)*acd40(31)
      acd40(63)=acd40(34)*acd40(33)
      acd40(64)=acd40(36)*acd40(35)
      acd40(65)=acd40(38)*acd40(37)
      acd40(66)=acd40(40)*acd40(39)
      acd40(67)=acd40(42)*acd40(41)
      acd40(68)=acd40(44)*acd40(43)
      acd40(69)=acd40(46)*acd40(45)
      acd40(47)=acd40(69)+acd40(68)+acd40(67)+acd40(66)+acd40(65)+acd40(64)+acd&
      &40(63)+acd40(62)+acd40(61)+acd40(60)+acd40(59)+acd40(58)+acd40(57)+acd40&
      &(56)+acd40(55)+acd40(54)+acd40(53)+acd40(52)+acd40(51)+acd40(50)+acd40(4&
      &9)+acd40(47)+acd40(48)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd40(47)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd40h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(94) :: acd40
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd40(1)=dotproduct(k2,ninjaA1)
      acd40(2)=abb40(27)
      acd40(3)=dotproduct(l4,ninjaA1)
      acd40(4)=abb40(16)
      acd40(5)=dotproduct(ninjaA1,spvae2k2)
      acd40(6)=abb40(14)
      acd40(7)=dotproduct(ninjaA1,spvak2l4)
      acd40(8)=abb40(15)
      acd40(9)=dotproduct(ninjaA1,spvak2e2)
      acd40(10)=abb40(17)
      acd40(11)=dotproduct(ninjaA1,spvak2l5)
      acd40(12)=abb40(18)
      acd40(13)=dotproduct(ninjaA1,spvae2l4)
      acd40(14)=abb40(19)
      acd40(15)=dotproduct(ninjaA1,spvak2e1)
      acd40(16)=abb40(20)
      acd40(17)=dotproduct(ninjaA1,spvae2k1)
      acd40(18)=abb40(21)
      acd40(19)=dotproduct(ninjaA1,spvak2k1)
      acd40(20)=abb40(22)
      acd40(21)=dotproduct(ninjaA1,spvak1l4)
      acd40(22)=abb40(24)
      acd40(23)=dotproduct(ninjaA1,spvak1e2)
      acd40(24)=abb40(25)
      acd40(25)=dotproduct(ninjaA1,spval5l4)
      acd40(26)=abb40(29)
      acd40(27)=dotproduct(ninjaA1,spvae2e1)
      acd40(28)=abb40(30)
      acd40(29)=dotproduct(ninjaA1,spval4k2)
      acd40(30)=abb40(31)
      acd40(31)=dotproduct(ninjaA1,spvae1e2)
      acd40(32)=abb40(32)
      acd40(33)=dotproduct(ninjaA1,spval4e1)
      acd40(34)=abb40(33)
      acd40(35)=dotproduct(ninjaA1,spvae2l5)
      acd40(36)=abb40(36)
      acd40(37)=dotproduct(ninjaA1,spval4l5)
      acd40(38)=abb40(37)
      acd40(39)=dotproduct(ninjaA1,spval4k1)
      acd40(40)=abb40(39)
      acd40(41)=dotproduct(ninjaA1,spval5e2)
      acd40(42)=abb40(44)
      acd40(43)=dotproduct(ninjaA1,spvae1l4)
      acd40(44)=abb40(46)
      acd40(45)=dotproduct(ninjaA1,spval4e2)
      acd40(46)=abb40(50)
      acd40(47)=dotproduct(k2,ninjaA0)
      acd40(48)=dotproduct(l4,ninjaA0)
      acd40(49)=dotproduct(ninjaA0,spvae2k2)
      acd40(50)=dotproduct(ninjaA0,spvak2l4)
      acd40(51)=dotproduct(ninjaA0,spvak2e2)
      acd40(52)=dotproduct(ninjaA0,spvak2l5)
      acd40(53)=dotproduct(ninjaA0,spvae2l4)
      acd40(54)=dotproduct(ninjaA0,spvak2e1)
      acd40(55)=dotproduct(ninjaA0,spvae2k1)
      acd40(56)=dotproduct(ninjaA0,spvak2k1)
      acd40(57)=dotproduct(ninjaA0,spvak1l4)
      acd40(58)=dotproduct(ninjaA0,spvak1e2)
      acd40(59)=dotproduct(ninjaA0,spval5l4)
      acd40(60)=dotproduct(ninjaA0,spvae2e1)
      acd40(61)=dotproduct(ninjaA0,spval4k2)
      acd40(62)=dotproduct(ninjaA0,spvae1e2)
      acd40(63)=dotproduct(ninjaA0,spval4e1)
      acd40(64)=dotproduct(ninjaA0,spvae2l5)
      acd40(65)=dotproduct(ninjaA0,spval4l5)
      acd40(66)=dotproduct(ninjaA0,spval4k1)
      acd40(67)=dotproduct(ninjaA0,spval5e2)
      acd40(68)=dotproduct(ninjaA0,spvae1l4)
      acd40(69)=dotproduct(ninjaA0,spval4e2)
      acd40(70)=abb40(23)
      acd40(71)=acd40(1)*acd40(2)
      acd40(72)=acd40(3)*acd40(4)
      acd40(73)=acd40(5)*acd40(6)
      acd40(74)=acd40(7)*acd40(8)
      acd40(75)=acd40(9)*acd40(10)
      acd40(76)=acd40(11)*acd40(12)
      acd40(77)=acd40(13)*acd40(14)
      acd40(78)=acd40(15)*acd40(16)
      acd40(79)=acd40(17)*acd40(18)
      acd40(80)=acd40(19)*acd40(20)
      acd40(81)=acd40(21)*acd40(22)
      acd40(82)=acd40(23)*acd40(24)
      acd40(83)=acd40(25)*acd40(26)
      acd40(84)=acd40(27)*acd40(28)
      acd40(85)=acd40(29)*acd40(30)
      acd40(86)=acd40(31)*acd40(32)
      acd40(87)=acd40(33)*acd40(34)
      acd40(88)=acd40(35)*acd40(36)
      acd40(89)=acd40(37)*acd40(38)
      acd40(90)=acd40(39)*acd40(40)
      acd40(91)=acd40(41)*acd40(42)
      acd40(92)=acd40(43)*acd40(44)
      acd40(93)=acd40(45)*acd40(46)
      acd40(71)=acd40(93)+acd40(92)+acd40(91)+acd40(90)+acd40(89)+acd40(88)+acd&
      &40(87)+acd40(86)+acd40(85)+acd40(84)+acd40(83)+acd40(82)+acd40(81)+acd40&
      &(80)+acd40(79)+acd40(78)+acd40(77)+acd40(76)+acd40(75)+acd40(74)+acd40(7&
      &3)+acd40(71)+acd40(72)
      acd40(72)=acd40(47)*acd40(2)
      acd40(73)=acd40(48)*acd40(4)
      acd40(74)=acd40(49)*acd40(6)
      acd40(75)=acd40(50)*acd40(8)
      acd40(76)=acd40(51)*acd40(10)
      acd40(77)=acd40(52)*acd40(12)
      acd40(78)=acd40(53)*acd40(14)
      acd40(79)=acd40(54)*acd40(16)
      acd40(80)=acd40(55)*acd40(18)
      acd40(81)=acd40(56)*acd40(20)
      acd40(82)=acd40(57)*acd40(22)
      acd40(83)=acd40(58)*acd40(24)
      acd40(84)=acd40(59)*acd40(26)
      acd40(85)=acd40(60)*acd40(28)
      acd40(86)=acd40(61)*acd40(30)
      acd40(87)=acd40(62)*acd40(32)
      acd40(88)=acd40(63)*acd40(34)
      acd40(89)=acd40(64)*acd40(36)
      acd40(90)=acd40(65)*acd40(38)
      acd40(91)=acd40(66)*acd40(40)
      acd40(92)=acd40(67)*acd40(42)
      acd40(93)=acd40(68)*acd40(44)
      acd40(94)=acd40(69)*acd40(46)
      acd40(72)=acd40(70)+acd40(94)+acd40(93)+acd40(92)+acd40(91)+acd40(90)+acd&
      &40(89)+acd40(88)+acd40(87)+acd40(86)+acd40(85)+acd40(84)+acd40(83)+acd40&
      &(82)+acd40(81)+acd40(80)+acd40(79)+acd40(78)+acd40(77)+acd40(76)+acd40(7&
      &5)+acd40(74)+acd40(72)+acd40(73)
      brack(ninjaidxt0x0mu0)=acd40(72)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd40(71)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d40h0_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd40h0
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
end module     p2_gg_httbar_d40h0l132
