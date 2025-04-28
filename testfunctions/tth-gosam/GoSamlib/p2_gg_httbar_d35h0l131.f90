module     p2_gg_httbar_d35h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d35h0l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd35h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(31) :: acd35
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd35(1)=dotproduct(k2,ninjaE3)
      acd35(2)=dotproduct(e1,ninjaE3)
      acd35(3)=abb35(17)
      acd35(4)=dotproduct(ninjaE3,spval5k2)
      acd35(5)=abb35(10)
      acd35(6)=dotproduct(ninjaE3,spval4l5)
      acd35(7)=abb35(22)
      acd35(8)=dotproduct(ninjaE3,spval5l3)
      acd35(9)=abb35(27)
      acd35(10)=dotproduct(ninjaE3,spvae2k2)
      acd35(11)=abb35(28)
      acd35(12)=dotproduct(ninjaE3,spvak2l3)
      acd35(13)=abb35(35)
      acd35(14)=dotproduct(ninjaE3,spval4e2)
      acd35(15)=abb35(38)
      acd35(16)=dotproduct(ninjaE3,spval3l5)
      acd35(17)=abb35(42)
      acd35(18)=dotproduct(ninjaE3,spvae2l3)
      acd35(19)=abb35(46)
      acd35(20)=dotproduct(ninjaE3,spval3e2)
      acd35(21)=abb35(47)
      acd35(22)=acd35(3)*acd35(1)
      acd35(23)=acd35(5)*acd35(4)
      acd35(24)=acd35(7)*acd35(6)
      acd35(25)=acd35(9)*acd35(8)
      acd35(26)=acd35(11)*acd35(10)
      acd35(27)=acd35(13)*acd35(12)
      acd35(28)=acd35(15)*acd35(14)
      acd35(29)=acd35(17)*acd35(16)
      acd35(30)=acd35(19)*acd35(18)
      acd35(31)=acd35(21)*acd35(20)
      acd35(22)=acd35(31)+acd35(30)+acd35(29)+acd35(28)+acd35(27)+acd35(26)+acd&
      &35(25)+acd35(24)+acd35(22)+acd35(23)
      acd35(22)=acd35(2)*acd35(22)
      brack(ninjaidxt2mu0)=acd35(22)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd35h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(139) :: acd35
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd35(1)=dotproduct(k2,ninjaE3)
      acd35(2)=dotproduct(e1,ninjaE4)
      acd35(3)=abb35(17)
      acd35(4)=dotproduct(k2,ninjaE4)
      acd35(5)=dotproduct(e1,ninjaE3)
      acd35(6)=dotproduct(ninjaE4,spval5k2)
      acd35(7)=abb35(10)
      acd35(8)=dotproduct(ninjaE4,spval3l5)
      acd35(9)=abb35(42)
      acd35(10)=dotproduct(ninjaE4,spval4l5)
      acd35(11)=abb35(22)
      acd35(12)=dotproduct(ninjaE4,spval5l3)
      acd35(13)=abb35(27)
      acd35(14)=dotproduct(ninjaE4,spvae2k2)
      acd35(15)=abb35(28)
      acd35(16)=dotproduct(ninjaE4,spvak2l3)
      acd35(17)=abb35(35)
      acd35(18)=dotproduct(ninjaE4,spval4e2)
      acd35(19)=abb35(38)
      acd35(20)=dotproduct(ninjaE4,spvae2l3)
      acd35(21)=abb35(46)
      acd35(22)=dotproduct(ninjaE4,spval3e2)
      acd35(23)=abb35(47)
      acd35(24)=dotproduct(ninjaE3,spval5k2)
      acd35(25)=dotproduct(ninjaE3,spval3l5)
      acd35(26)=dotproduct(ninjaE3,spval4l5)
      acd35(27)=dotproduct(ninjaE3,spval5l3)
      acd35(28)=dotproduct(ninjaE3,spvae2k2)
      acd35(29)=dotproduct(ninjaE3,spvak2l3)
      acd35(30)=dotproduct(ninjaE3,spval4e2)
      acd35(31)=dotproduct(ninjaE3,spvae2l3)
      acd35(32)=dotproduct(ninjaE3,spval3e2)
      acd35(33)=abb35(37)
      acd35(34)=dotproduct(k2,ninjaA)
      acd35(35)=dotproduct(e1,ninjaA)
      acd35(36)=abb35(16)
      acd35(37)=dotproduct(l5,ninjaE3)
      acd35(38)=abb35(12)
      acd35(39)=dotproduct(ninjaA,spval5k2)
      acd35(40)=dotproduct(ninjaA,spval3l5)
      acd35(41)=dotproduct(ninjaA,spval4l5)
      acd35(42)=dotproduct(ninjaA,spval5l3)
      acd35(43)=dotproduct(ninjaA,spvae2k2)
      acd35(44)=dotproduct(ninjaA,spvak2l3)
      acd35(45)=dotproduct(ninjaA,spval4e2)
      acd35(46)=dotproduct(ninjaA,spvae2l3)
      acd35(47)=dotproduct(ninjaA,spval3e2)
      acd35(48)=abb35(9)
      acd35(49)=dotproduct(ninjaA,ninjaE3)
      acd35(50)=abb35(36)
      acd35(51)=dotproduct(ninjaE3,spval3e1)
      acd35(52)=abb35(11)
      acd35(53)=abb35(13)
      acd35(54)=dotproduct(ninjaE3,spvae1k1)
      acd35(55)=abb35(14)
      acd35(56)=dotproduct(ninjaE3,spvae2e1)
      acd35(57)=abb35(18)
      acd35(58)=dotproduct(ninjaE3,spvae1e2)
      acd35(59)=abb35(19)
      acd35(60)=abb35(20)
      acd35(61)=dotproduct(ninjaE3,spvak2e2)
      acd35(62)=abb35(21)
      acd35(63)=dotproduct(ninjaE3,spvae2l5)
      acd35(64)=abb35(23)
      acd35(65)=dotproduct(ninjaE3,spvak1e1)
      acd35(66)=abb35(24)
      acd35(67)=dotproduct(ninjaE3,spval3k2)
      acd35(68)=abb35(25)
      acd35(69)=abb35(26)
      acd35(70)=abb35(31)
      acd35(71)=dotproduct(ninjaE3,spval5e1)
      acd35(72)=abb35(29)
      acd35(73)=dotproduct(ninjaE3,spval5e2)
      acd35(74)=abb35(30)
      acd35(75)=dotproduct(ninjaE3,spvak2e1)
      acd35(76)=abb35(32)
      acd35(77)=dotproduct(ninjaE3,spvak2l5)
      acd35(78)=abb35(33)
      acd35(79)=dotproduct(ninjaE3,spvae1l5)
      acd35(80)=abb35(34)
      acd35(81)=abb35(43)
      acd35(82)=dotproduct(ninjaE3,spval4e1)
      acd35(83)=abb35(40)
      acd35(84)=dotproduct(ninjaE3,spval4k2)
      acd35(85)=abb35(41)
      acd35(86)=dotproduct(ninjaE3,spvae1l3)
      acd35(87)=abb35(49)
      acd35(88)=dotproduct(ninjaE3,spvae1k2)
      acd35(89)=abb35(76)
      acd35(90)=dotproduct(l5,ninjaA)
      acd35(91)=dotproduct(ninjaA,ninjaA)
      acd35(92)=dotproduct(ninjaA,spval3e1)
      acd35(93)=dotproduct(ninjaA,spvae1k1)
      acd35(94)=dotproduct(ninjaA,spvae2e1)
      acd35(95)=dotproduct(ninjaA,spvae1e2)
      acd35(96)=dotproduct(ninjaA,spvak2e2)
      acd35(97)=dotproduct(ninjaA,spvae2l5)
      acd35(98)=dotproduct(ninjaA,spvak1e1)
      acd35(99)=dotproduct(ninjaA,spval3k2)
      acd35(100)=dotproduct(ninjaA,spval5e1)
      acd35(101)=dotproduct(ninjaA,spval5e2)
      acd35(102)=dotproduct(ninjaA,spvak2e1)
      acd35(103)=dotproduct(ninjaA,spvak2l5)
      acd35(104)=dotproduct(ninjaA,spvae1l5)
      acd35(105)=dotproduct(ninjaA,spval4e1)
      acd35(106)=dotproduct(ninjaA,spval4k2)
      acd35(107)=dotproduct(ninjaA,spvae1l3)
      acd35(108)=dotproduct(ninjaA,spvae1k2)
      acd35(109)=abb35(15)
      acd35(110)=acd35(23)*acd35(22)
      acd35(111)=acd35(21)*acd35(20)
      acd35(112)=acd35(19)*acd35(18)
      acd35(113)=acd35(17)*acd35(16)
      acd35(114)=acd35(15)*acd35(14)
      acd35(115)=acd35(13)*acd35(12)
      acd35(116)=acd35(11)*acd35(10)
      acd35(117)=acd35(9)*acd35(8)
      acd35(118)=acd35(7)*acd35(6)
      acd35(119)=acd35(3)*acd35(4)
      acd35(110)=acd35(110)+acd35(114)+acd35(115)+acd35(111)+acd35(112)+acd35(1&
      &13)+acd35(116)+acd35(117)+acd35(118)+acd35(119)
      acd35(110)=acd35(110)*acd35(5)
      acd35(111)=acd35(23)*acd35(32)
      acd35(112)=acd35(21)*acd35(31)
      acd35(113)=acd35(19)*acd35(30)
      acd35(114)=acd35(17)*acd35(29)
      acd35(115)=acd35(15)*acd35(28)
      acd35(116)=acd35(13)*acd35(27)
      acd35(117)=acd35(11)*acd35(26)
      acd35(118)=acd35(9)*acd35(25)
      acd35(119)=acd35(7)*acd35(24)
      acd35(120)=acd35(3)*acd35(1)
      acd35(111)=acd35(116)+acd35(115)+acd35(114)+acd35(113)+acd35(111)+acd35(1&
      &12)+acd35(117)+acd35(118)+acd35(119)+acd35(120)
      acd35(112)=acd35(111)*acd35(2)
      acd35(110)=acd35(110)+acd35(112)+acd35(33)
      acd35(111)=acd35(35)*acd35(111)
      acd35(112)=acd35(23)*acd35(47)
      acd35(113)=acd35(21)*acd35(46)
      acd35(114)=acd35(19)*acd35(45)
      acd35(115)=acd35(17)*acd35(44)
      acd35(116)=acd35(15)*acd35(43)
      acd35(117)=acd35(13)*acd35(42)
      acd35(118)=acd35(11)*acd35(41)
      acd35(119)=acd35(9)*acd35(40)
      acd35(120)=acd35(7)*acd35(39)
      acd35(121)=acd35(3)*acd35(34)
      acd35(112)=acd35(115)+acd35(116)+acd35(117)+acd35(118)+acd35(112)+acd35(1&
      &13)+acd35(114)+acd35(119)+acd35(120)+acd35(121)+acd35(48)
      acd35(113)=acd35(5)*acd35(112)
      acd35(114)=acd35(89)*acd35(88)
      acd35(115)=acd35(87)*acd35(86)
      acd35(116)=acd35(85)*acd35(84)
      acd35(117)=acd35(83)*acd35(82)
      acd35(118)=acd35(80)*acd35(79)
      acd35(119)=acd35(78)*acd35(77)
      acd35(120)=acd35(76)*acd35(75)
      acd35(121)=acd35(74)*acd35(73)
      acd35(122)=acd35(72)*acd35(71)
      acd35(123)=acd35(68)*acd35(67)
      acd35(124)=acd35(66)*acd35(65)
      acd35(125)=acd35(64)*acd35(63)
      acd35(126)=acd35(62)*acd35(61)
      acd35(127)=acd35(59)*acd35(58)
      acd35(128)=acd35(57)*acd35(56)
      acd35(129)=acd35(55)*acd35(54)
      acd35(130)=acd35(52)*acd35(51)
      acd35(131)=acd35(38)*acd35(37)
      acd35(132)=acd35(33)*acd35(49)
      acd35(133)=acd35(29)*acd35(81)
      acd35(134)=acd35(28)*acd35(70)
      acd35(135)=acd35(27)*acd35(69)
      acd35(136)=acd35(26)*acd35(60)
      acd35(137)=acd35(25)*acd35(53)
      acd35(138)=acd35(24)*acd35(50)
      acd35(139)=acd35(1)*acd35(36)
      acd35(111)=acd35(113)+acd35(111)+acd35(139)+acd35(138)+acd35(137)+acd35(1&
      &36)+acd35(135)+acd35(134)+acd35(133)+2.0_ki*acd35(132)+acd35(131)+acd35(&
      &130)+acd35(129)+acd35(128)+acd35(127)+acd35(126)+acd35(125)+acd35(124)+a&
      &cd35(123)+acd35(122)+acd35(121)+acd35(120)+acd35(119)+acd35(118)+acd35(1&
      &17)+acd35(116)+acd35(114)+acd35(115)
      acd35(113)=ninjaP*acd35(110)
      acd35(112)=acd35(35)*acd35(112)
      acd35(114)=acd35(89)*acd35(108)
      acd35(115)=acd35(87)*acd35(107)
      acd35(116)=acd35(85)*acd35(106)
      acd35(117)=acd35(83)*acd35(105)
      acd35(118)=acd35(80)*acd35(104)
      acd35(119)=acd35(78)*acd35(103)
      acd35(120)=acd35(76)*acd35(102)
      acd35(121)=acd35(74)*acd35(101)
      acd35(122)=acd35(72)*acd35(100)
      acd35(123)=acd35(68)*acd35(99)
      acd35(124)=acd35(66)*acd35(98)
      acd35(125)=acd35(64)*acd35(97)
      acd35(126)=acd35(62)*acd35(96)
      acd35(127)=acd35(59)*acd35(95)
      acd35(128)=acd35(57)*acd35(94)
      acd35(129)=acd35(55)*acd35(93)
      acd35(130)=acd35(52)*acd35(92)
      acd35(131)=acd35(38)*acd35(90)
      acd35(132)=acd35(44)*acd35(81)
      acd35(133)=acd35(43)*acd35(70)
      acd35(134)=acd35(42)*acd35(69)
      acd35(135)=acd35(41)*acd35(60)
      acd35(136)=acd35(40)*acd35(53)
      acd35(137)=acd35(39)*acd35(50)
      acd35(138)=acd35(34)*acd35(36)
      acd35(139)=acd35(33)*acd35(91)
      acd35(112)=acd35(112)+acd35(139)+acd35(138)+acd35(137)+acd35(136)+acd35(1&
      &35)+acd35(134)+acd35(133)+acd35(132)+acd35(131)+acd35(130)+acd35(129)+ac&
      &d35(128)+acd35(127)+acd35(126)+acd35(125)+acd35(124)+acd35(123)+acd35(12&
      &2)+acd35(121)+acd35(120)+acd35(119)+acd35(118)+acd35(117)+acd35(116)+acd&
      &35(115)+acd35(109)+acd35(114)+acd35(113)
      brack(ninjaidxt1mu0)=acd35(111)
      brack(ninjaidxt0mu0)=acd35(112)
      brack(ninjaidxt0mu2)=acd35(110)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d35h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd35h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d35h0l131
