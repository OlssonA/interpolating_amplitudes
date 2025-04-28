module     p2_gg_httbar_d42h12l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d42h12l132.f90
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
      use p2_gg_httbar_abbrevd42h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(87) :: acd42
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd42(1)=dotproduct(k1,ninjaE3)
      acd42(2)=abb42(19)
      acd42(3)=dotproduct(k2,ninjaE3)
      acd42(4)=abb42(15)
      acd42(5)=dotproduct(l5,ninjaE3)
      acd42(6)=abb42(37)
      acd42(7)=dotproduct(ninjaE3,spvak1e2)
      acd42(8)=abb42(14)
      acd42(9)=dotproduct(ninjaE3,spvae1k2)
      acd42(10)=abb42(16)
      acd42(11)=dotproduct(ninjaE3,spvak2k1)
      acd42(12)=abb42(17)
      acd42(13)=dotproduct(ninjaE3,spvak1l5)
      acd42(14)=abb42(20)
      acd42(15)=dotproduct(ninjaE3,spvak1e1)
      acd42(16)=abb42(22)
      acd42(17)=dotproduct(ninjaE3,spval4k1)
      acd42(18)=abb42(23)
      acd42(19)=dotproduct(ninjaE3,spval5k1)
      acd42(20)=abb42(24)
      acd42(21)=dotproduct(ninjaE3,spvae1e2)
      acd42(22)=abb42(25)
      acd42(23)=dotproduct(ninjaE3,spvae2e1)
      acd42(24)=abb42(26)
      acd42(25)=dotproduct(ninjaE3,spvak1k2)
      acd42(26)=abb42(27)
      acd42(27)=dotproduct(ninjaE3,spval4l5)
      acd42(28)=abb42(30)
      acd42(29)=dotproduct(ninjaE3,spvak1l4)
      acd42(30)=abb42(31)
      acd42(31)=dotproduct(ninjaE3,spval5e2)
      acd42(32)=abb42(32)
      acd42(33)=dotproduct(ninjaE3,spval4k2)
      acd42(34)=abb42(35)
      acd42(35)=dotproduct(ninjaE3,spvak2l5)
      acd42(36)=abb42(39)
      acd42(37)=dotproduct(ninjaE3,spvak2e1)
      acd42(38)=abb42(40)
      acd42(39)=dotproduct(ninjaE3,spvae2l5)
      acd42(40)=abb42(45)
      acd42(41)=dotproduct(ninjaE3,spvae2k2)
      acd42(42)=abb42(46)
      acd42(43)=dotproduct(ninjaE3,spvae1k1)
      acd42(44)=abb42(47)
      acd42(45)=dotproduct(ninjaE3,spvae2k1)
      acd42(46)=abb42(48)
      acd42(47)=dotproduct(ninjaE3,spvae1l5)
      acd42(48)=abb42(49)
      acd42(49)=dotproduct(ninjaE3,spval5e1)
      acd42(50)=abb42(53)
      acd42(51)=dotproduct(ninjaE3,spval5l4)
      acd42(52)=abb42(55)
      acd42(53)=dotproduct(ninjaE3,spval5k2)
      acd42(54)=abb42(56)
      acd42(55)=dotproduct(ninjaE3,spvae1l4)
      acd42(56)=abb42(58)
      acd42(57)=dotproduct(ninjaE3,spval4e1)
      acd42(58)=abb42(59)
      acd42(59)=acd42(2)*acd42(1)
      acd42(60)=acd42(4)*acd42(3)
      acd42(61)=acd42(6)*acd42(5)
      acd42(62)=acd42(8)*acd42(7)
      acd42(63)=acd42(10)*acd42(9)
      acd42(64)=acd42(12)*acd42(11)
      acd42(65)=acd42(14)*acd42(13)
      acd42(66)=acd42(16)*acd42(15)
      acd42(67)=acd42(18)*acd42(17)
      acd42(68)=acd42(20)*acd42(19)
      acd42(69)=acd42(22)*acd42(21)
      acd42(70)=acd42(24)*acd42(23)
      acd42(71)=acd42(26)*acd42(25)
      acd42(72)=acd42(28)*acd42(27)
      acd42(73)=acd42(30)*acd42(29)
      acd42(74)=acd42(32)*acd42(31)
      acd42(75)=acd42(34)*acd42(33)
      acd42(76)=acd42(36)*acd42(35)
      acd42(77)=acd42(38)*acd42(37)
      acd42(78)=acd42(40)*acd42(39)
      acd42(79)=acd42(42)*acd42(41)
      acd42(80)=acd42(44)*acd42(43)
      acd42(81)=acd42(46)*acd42(45)
      acd42(82)=acd42(48)*acd42(47)
      acd42(83)=acd42(50)*acd42(49)
      acd42(84)=acd42(52)*acd42(51)
      acd42(85)=acd42(54)*acd42(53)
      acd42(86)=acd42(56)*acd42(55)
      acd42(87)=acd42(58)*acd42(57)
      acd42(59)=acd42(87)+acd42(86)+acd42(85)+acd42(84)+acd42(83)+acd42(82)+acd&
      &42(81)+acd42(80)+acd42(79)+acd42(78)+acd42(77)+acd42(76)+acd42(75)+acd42&
      &(74)+acd42(73)+acd42(72)+acd42(71)+acd42(70)+acd42(69)+acd42(68)+acd42(6&
      &7)+acd42(66)+acd42(65)+acd42(64)+acd42(63)+acd42(62)+acd42(61)+acd42(59)&
      &+acd42(60)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd42(59)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd42h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(118) :: acd42
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd42(1)=dotproduct(k1,ninjaA1)
      acd42(2)=abb42(19)
      acd42(3)=dotproduct(k2,ninjaA1)
      acd42(4)=abb42(15)
      acd42(5)=dotproduct(l5,ninjaA1)
      acd42(6)=abb42(37)
      acd42(7)=dotproduct(ninjaA1,spvak1e2)
      acd42(8)=abb42(14)
      acd42(9)=dotproduct(ninjaA1,spvae1k2)
      acd42(10)=abb42(16)
      acd42(11)=dotproduct(ninjaA1,spvak2k1)
      acd42(12)=abb42(17)
      acd42(13)=dotproduct(ninjaA1,spvak1l5)
      acd42(14)=abb42(20)
      acd42(15)=dotproduct(ninjaA1,spvak1e1)
      acd42(16)=abb42(22)
      acd42(17)=dotproduct(ninjaA1,spval4k1)
      acd42(18)=abb42(23)
      acd42(19)=dotproduct(ninjaA1,spval5k1)
      acd42(20)=abb42(24)
      acd42(21)=dotproduct(ninjaA1,spvae1e2)
      acd42(22)=abb42(25)
      acd42(23)=dotproduct(ninjaA1,spvae2e1)
      acd42(24)=abb42(26)
      acd42(25)=dotproduct(ninjaA1,spvak1k2)
      acd42(26)=abb42(27)
      acd42(27)=dotproduct(ninjaA1,spval4l5)
      acd42(28)=abb42(30)
      acd42(29)=dotproduct(ninjaA1,spvak1l4)
      acd42(30)=abb42(31)
      acd42(31)=dotproduct(ninjaA1,spval5e2)
      acd42(32)=abb42(32)
      acd42(33)=dotproduct(ninjaA1,spval4k2)
      acd42(34)=abb42(35)
      acd42(35)=dotproduct(ninjaA1,spvak2l5)
      acd42(36)=abb42(39)
      acd42(37)=dotproduct(ninjaA1,spvak2e1)
      acd42(38)=abb42(40)
      acd42(39)=dotproduct(ninjaA1,spvae2l5)
      acd42(40)=abb42(45)
      acd42(41)=dotproduct(ninjaA1,spvae2k2)
      acd42(42)=abb42(46)
      acd42(43)=dotproduct(ninjaA1,spvae1k1)
      acd42(44)=abb42(47)
      acd42(45)=dotproduct(ninjaA1,spvae2k1)
      acd42(46)=abb42(48)
      acd42(47)=dotproduct(ninjaA1,spvae1l5)
      acd42(48)=abb42(49)
      acd42(49)=dotproduct(ninjaA1,spval5e1)
      acd42(50)=abb42(53)
      acd42(51)=dotproduct(ninjaA1,spval5l4)
      acd42(52)=abb42(55)
      acd42(53)=dotproduct(ninjaA1,spval5k2)
      acd42(54)=abb42(56)
      acd42(55)=dotproduct(ninjaA1,spvae1l4)
      acd42(56)=abb42(58)
      acd42(57)=dotproduct(ninjaA1,spval4e1)
      acd42(58)=abb42(59)
      acd42(59)=dotproduct(k1,ninjaA0)
      acd42(60)=dotproduct(k2,ninjaA0)
      acd42(61)=dotproduct(l5,ninjaA0)
      acd42(62)=dotproduct(ninjaA0,spvak1e2)
      acd42(63)=dotproduct(ninjaA0,spvae1k2)
      acd42(64)=dotproduct(ninjaA0,spvak2k1)
      acd42(65)=dotproduct(ninjaA0,spvak1l5)
      acd42(66)=dotproduct(ninjaA0,spvak1e1)
      acd42(67)=dotproduct(ninjaA0,spval4k1)
      acd42(68)=dotproduct(ninjaA0,spval5k1)
      acd42(69)=dotproduct(ninjaA0,spvae1e2)
      acd42(70)=dotproduct(ninjaA0,spvae2e1)
      acd42(71)=dotproduct(ninjaA0,spvak1k2)
      acd42(72)=dotproduct(ninjaA0,spval4l5)
      acd42(73)=dotproduct(ninjaA0,spvak1l4)
      acd42(74)=dotproduct(ninjaA0,spval5e2)
      acd42(75)=dotproduct(ninjaA0,spval4k2)
      acd42(76)=dotproduct(ninjaA0,spvak2l5)
      acd42(77)=dotproduct(ninjaA0,spvak2e1)
      acd42(78)=dotproduct(ninjaA0,spvae2l5)
      acd42(79)=dotproduct(ninjaA0,spvae2k2)
      acd42(80)=dotproduct(ninjaA0,spvae1k1)
      acd42(81)=dotproduct(ninjaA0,spvae2k1)
      acd42(82)=dotproduct(ninjaA0,spvae1l5)
      acd42(83)=dotproduct(ninjaA0,spval5e1)
      acd42(84)=dotproduct(ninjaA0,spval5l4)
      acd42(85)=dotproduct(ninjaA0,spval5k2)
      acd42(86)=dotproduct(ninjaA0,spvae1l4)
      acd42(87)=dotproduct(ninjaA0,spval4e1)
      acd42(88)=abb42(21)
      acd42(89)=acd42(1)*acd42(2)
      acd42(90)=acd42(3)*acd42(4)
      acd42(91)=acd42(5)*acd42(6)
      acd42(92)=acd42(7)*acd42(8)
      acd42(93)=acd42(9)*acd42(10)
      acd42(94)=acd42(11)*acd42(12)
      acd42(95)=acd42(13)*acd42(14)
      acd42(96)=acd42(15)*acd42(16)
      acd42(97)=acd42(17)*acd42(18)
      acd42(98)=acd42(19)*acd42(20)
      acd42(99)=acd42(21)*acd42(22)
      acd42(100)=acd42(23)*acd42(24)
      acd42(101)=acd42(25)*acd42(26)
      acd42(102)=acd42(27)*acd42(28)
      acd42(103)=acd42(29)*acd42(30)
      acd42(104)=acd42(31)*acd42(32)
      acd42(105)=acd42(33)*acd42(34)
      acd42(106)=acd42(35)*acd42(36)
      acd42(107)=acd42(37)*acd42(38)
      acd42(108)=acd42(39)*acd42(40)
      acd42(109)=acd42(41)*acd42(42)
      acd42(110)=acd42(43)*acd42(44)
      acd42(111)=acd42(45)*acd42(46)
      acd42(112)=acd42(47)*acd42(48)
      acd42(113)=acd42(49)*acd42(50)
      acd42(114)=acd42(51)*acd42(52)
      acd42(115)=acd42(53)*acd42(54)
      acd42(116)=acd42(55)*acd42(56)
      acd42(117)=acd42(57)*acd42(58)
      acd42(89)=acd42(117)+acd42(116)+acd42(115)+acd42(114)+acd42(113)+acd42(11&
      &2)+acd42(111)+acd42(110)+acd42(109)+acd42(108)+acd42(107)+acd42(106)+acd&
      &42(105)+acd42(104)+acd42(103)+acd42(102)+acd42(101)+acd42(100)+acd42(99)&
      &+acd42(98)+acd42(97)+acd42(96)+acd42(95)+acd42(94)+acd42(93)+acd42(92)+a&
      &cd42(91)+acd42(89)+acd42(90)
      acd42(90)=acd42(59)*acd42(2)
      acd42(91)=acd42(60)*acd42(4)
      acd42(92)=acd42(61)*acd42(6)
      acd42(93)=acd42(62)*acd42(8)
      acd42(94)=acd42(63)*acd42(10)
      acd42(95)=acd42(64)*acd42(12)
      acd42(96)=acd42(65)*acd42(14)
      acd42(97)=acd42(66)*acd42(16)
      acd42(98)=acd42(67)*acd42(18)
      acd42(99)=acd42(68)*acd42(20)
      acd42(100)=acd42(69)*acd42(22)
      acd42(101)=acd42(70)*acd42(24)
      acd42(102)=acd42(71)*acd42(26)
      acd42(103)=acd42(72)*acd42(28)
      acd42(104)=acd42(73)*acd42(30)
      acd42(105)=acd42(74)*acd42(32)
      acd42(106)=acd42(75)*acd42(34)
      acd42(107)=acd42(76)*acd42(36)
      acd42(108)=acd42(77)*acd42(38)
      acd42(109)=acd42(78)*acd42(40)
      acd42(110)=acd42(79)*acd42(42)
      acd42(111)=acd42(80)*acd42(44)
      acd42(112)=acd42(81)*acd42(46)
      acd42(113)=acd42(82)*acd42(48)
      acd42(114)=acd42(83)*acd42(50)
      acd42(115)=acd42(84)*acd42(52)
      acd42(116)=acd42(85)*acd42(54)
      acd42(117)=acd42(86)*acd42(56)
      acd42(118)=acd42(87)*acd42(58)
      acd42(90)=acd42(88)+acd42(118)+acd42(117)+acd42(116)+acd42(115)+acd42(114&
      &)+acd42(113)+acd42(112)+acd42(111)+acd42(110)+acd42(109)+acd42(108)+acd4&
      &2(107)+acd42(106)+acd42(105)+acd42(104)+acd42(103)+acd42(102)+acd42(101)&
      &+acd42(100)+acd42(99)+acd42(98)+acd42(97)+acd42(96)+acd42(95)+acd42(94)+&
      &acd42(93)+acd42(92)+acd42(90)+acd42(91)
      brack(ninjaidxt0x0mu0)=acd42(90)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd42(89)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d42h12_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd42h12
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
end module     p2_gg_httbar_d42h12l132
