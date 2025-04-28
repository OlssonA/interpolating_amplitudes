module     p2_gg_httbar_d257h4l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d257h4l131.f90
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
      use p2_gg_httbar_abbrevd257h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd257
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd257h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(97) :: acd257
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd257(1)=dotproduct(ninjaE3,spvae2k2)
      acd257(2)=dotproduct(ninjaE3,spvae1e2)
      acd257(3)=abb257(7)
      acd257(4)=dotproduct(ninjaE3,spvae2l4)
      acd257(5)=abb257(41)
      acd257(6)=dotproduct(ninjaE3,spval5e2)
      acd257(7)=dotproduct(ninjaE3,spvae2e1)
      acd257(8)=abb257(11)
      acd257(9)=dotproduct(ninjaE3,spvak2e2)
      acd257(10)=abb257(15)
      acd257(11)=dotproduct(k2,ninjaE3)
      acd257(12)=abb257(28)
      acd257(13)=abb257(32)
      acd257(14)=dotproduct(ninjaA,ninjaE3)
      acd257(15)=dotproduct(ninjaE3,spvak2l4)
      acd257(16)=abb257(24)
      acd257(17)=dotproduct(ninjaE3,spvak2e1)
      acd257(18)=abb257(52)
      acd257(19)=dotproduct(ninjaE3,spval3k2)
      acd257(20)=dotproduct(ninjaE3,spvae2l3)
      acd257(21)=dotproduct(ninjaE3,spval3l4)
      acd257(22)=dotproduct(ninjaE3,spval5k2)
      acd257(23)=abb257(40)
      acd257(24)=dotproduct(ninjaE3,spvae1l4)
      acd257(25)=abb257(66)
      acd257(26)=dotproduct(ninjaE3,spval5l3)
      acd257(27)=dotproduct(ninjaE3,spval3e2)
      acd257(28)=dotproduct(ninjaE3,spvak2l3)
      acd257(29)=dotproduct(k2,ninjaA)
      acd257(30)=dotproduct(ninjaA,spvae2k2)
      acd257(31)=dotproduct(ninjaA,spvae1e2)
      acd257(32)=dotproduct(ninjaA,spvae2e1)
      acd257(33)=dotproduct(ninjaA,spvak2e2)
      acd257(34)=abb257(23)
      acd257(35)=abb257(55)
      acd257(36)=dotproduct(ninjaA,ninjaA)
      acd257(37)=dotproduct(ninjaA,spval5e2)
      acd257(38)=dotproduct(ninjaA,spvae2l4)
      acd257(39)=abb257(51)
      acd257(40)=abb257(43)
      acd257(41)=abb257(20)
      acd257(42)=abb257(53)
      acd257(43)=dotproduct(ninjaA,spval3k2)
      acd257(44)=dotproduct(ninjaA,spvae2l3)
      acd257(45)=dotproduct(ninjaA,spval5k2)
      acd257(46)=dotproduct(ninjaA,spvae1l4)
      acd257(47)=dotproduct(ninjaA,spval5l3)
      acd257(48)=dotproduct(ninjaA,spval3e2)
      acd257(49)=dotproduct(ninjaA,spvak2l3)
      acd257(50)=dotproduct(ninjaA,spvak2l4)
      acd257(51)=dotproduct(ninjaA,spvak2e1)
      acd257(52)=dotproduct(ninjaA,spval3l4)
      acd257(53)=abb257(10)
      acd257(54)=abb257(34)
      acd257(55)=abb257(58)
      acd257(56)=abb257(54)
      acd257(57)=abb257(50)
      acd257(58)=abb257(57)
      acd257(59)=abb257(37)
      acd257(60)=abb257(59)
      acd257(61)=dotproduct(ninjaE3,spvak1k2)
      acd257(62)=abb257(48)
      acd257(63)=abb257(67)
      acd257(64)=abb257(8)
      acd257(65)=abb257(33)
      acd257(66)=abb257(12)
      acd257(67)=abb257(42)
      acd257(68)=abb257(22)
      acd257(69)=abb257(47)
      acd257(70)=abb257(56)
      acd257(71)=dotproduct(ninjaE3,spvak2k1)
      acd257(72)=abb257(39)
      acd257(73)=abb257(29)
      acd257(74)=acd257(8)*acd257(6)
      acd257(75)=acd257(9)*acd257(10)
      acd257(74)=acd257(74)-acd257(75)
      acd257(75)=-acd257(7)*acd257(74)
      acd257(76)=acd257(5)*acd257(4)
      acd257(77)=-acd257(1)*acd257(3)
      acd257(77)=acd257(76)+acd257(77)
      acd257(77)=acd257(2)*acd257(77)
      acd257(75)=acd257(75)+acd257(77)
      acd257(77)=acd257(11)*acd257(13)
      acd257(78)=acd257(22)*acd257(23)
      acd257(77)=acd257(77)+acd257(78)
      acd257(77)=acd257(77)*acd257(9)
      acd257(79)=acd257(10)*acd257(28)
      acd257(80)=acd257(8)*acd257(26)
      acd257(79)=acd257(79)-acd257(80)
      acd257(80)=-acd257(27)*acd257(79)
      acd257(81)=2.0_ki*acd257(14)
      acd257(82)=-acd257(74)*acd257(81)
      acd257(83)=acd257(6)*acd257(25)
      acd257(84)=acd257(24)*acd257(83)
      acd257(80)=acd257(82)+acd257(77)+acd257(84)+acd257(80)
      acd257(80)=acd257(7)*acd257(80)
      acd257(82)=acd257(16)*acd257(15)
      acd257(84)=acd257(11)*acd257(12)
      acd257(82)=acd257(82)-acd257(84)
      acd257(84)=acd257(17)*acd257(18)
      acd257(85)=acd257(81)*acd257(3)
      acd257(84)=acd257(84)+acd257(85)+acd257(82)
      acd257(85)=-acd257(1)*acd257(84)
      acd257(86)=acd257(21)*acd257(5)
      acd257(87)=acd257(19)*acd257(3)
      acd257(86)=acd257(86)-acd257(87)
      acd257(86)=acd257(20)*acd257(86)
      acd257(87)=acd257(81)*acd257(76)
      acd257(85)=acd257(85)+acd257(87)-acd257(86)
      acd257(85)=acd257(2)*acd257(85)
      acd257(80)=acd257(80)+acd257(85)
      acd257(85)=acd257(26)*acd257(48)
      acd257(87)=acd257(27)*acd257(47)
      acd257(88)=acd257(36)+ninjaP
      acd257(89)=-acd257(6)*acd257(88)
      acd257(85)=acd257(89)+acd257(85)+acd257(87)
      acd257(85)=acd257(8)*acd257(85)
      acd257(87)=acd257(23)*acd257(45)
      acd257(89)=acd257(13)*acd257(29)
      acd257(90)=acd257(10)*acd257(88)
      acd257(87)=acd257(90)+acd257(89)+acd257(69)+acd257(87)
      acd257(87)=acd257(9)*acd257(87)
      acd257(89)=-acd257(28)*acd257(48)
      acd257(90)=-acd257(27)*acd257(49)
      acd257(89)=acd257(89)+acd257(90)
      acd257(89)=acd257(10)*acd257(89)
      acd257(90)=acd257(10)*acd257(33)
      acd257(91)=-acd257(8)*acd257(37)
      acd257(90)=acd257(91)+acd257(41)+acd257(90)
      acd257(90)=acd257(90)*acd257(81)
      acd257(91)=-acd257(71)*acd257(72)
      acd257(92)=acd257(28)*acd257(70)
      acd257(93)=acd257(26)*acd257(67)
      acd257(78)=acd257(33)*acd257(78)
      acd257(94)=acd257(27)*acd257(68)
      acd257(95)=acd257(25)*acd257(37)
      acd257(95)=acd257(65)+acd257(95)
      acd257(95)=acd257(24)*acd257(95)
      acd257(96)=acd257(13)*acd257(33)
      acd257(96)=acd257(34)+acd257(96)
      acd257(96)=acd257(11)*acd257(96)
      acd257(97)=acd257(25)*acd257(46)
      acd257(97)=acd257(66)+acd257(97)
      acd257(97)=acd257(6)*acd257(97)
      acd257(78)=acd257(90)+acd257(87)+acd257(85)+acd257(89)+acd257(97)+acd257(&
      &96)+acd257(95)+acd257(94)+acd257(78)+acd257(93)+acd257(91)+acd257(92)
      acd257(78)=acd257(7)*acd257(78)
      acd257(85)=-acd257(18)*acd257(51)
      acd257(87)=-acd257(16)*acd257(50)
      acd257(89)=acd257(12)*acd257(29)
      acd257(90)=-acd257(3)*acd257(88)
      acd257(85)=acd257(90)+acd257(89)+acd257(87)+acd257(53)+acd257(85)
      acd257(85)=acd257(1)*acd257(85)
      acd257(87)=-acd257(21)*acd257(44)
      acd257(88)=acd257(4)*acd257(88)
      acd257(89)=-acd257(20)*acd257(52)
      acd257(87)=acd257(89)+acd257(87)+acd257(88)
      acd257(87)=acd257(5)*acd257(87)
      acd257(82)=-acd257(30)*acd257(82)
      acd257(88)=acd257(19)*acd257(44)
      acd257(89)=acd257(20)*acd257(43)
      acd257(88)=acd257(88)+acd257(89)
      acd257(88)=acd257(3)*acd257(88)
      acd257(89)=acd257(5)*acd257(38)
      acd257(90)=-acd257(3)*acd257(30)
      acd257(89)=acd257(90)+acd257(39)+acd257(89)
      acd257(89)=acd257(89)*acd257(81)
      acd257(90)=acd257(61)*acd257(62)
      acd257(91)=acd257(21)*acd257(59)
      acd257(92)=acd257(19)*acd257(55)
      acd257(93)=acd257(22)*acd257(57)
      acd257(94)=acd257(4)*acd257(60)
      acd257(95)=-acd257(18)*acd257(30)
      acd257(95)=acd257(58)+acd257(95)
      acd257(95)=acd257(17)*acd257(95)
      acd257(96)=acd257(20)*acd257(56)
      acd257(82)=acd257(85)+acd257(89)+acd257(88)+acd257(87)+acd257(96)+acd257(&
      &95)+acd257(94)+acd257(93)+acd257(92)+acd257(90)+acd257(91)+acd257(82)
      acd257(82)=acd257(2)*acd257(82)
      acd257(74)=-acd257(32)*acd257(74)
      acd257(85)=acd257(17)*acd257(42)
      acd257(87)=acd257(24)*acd257(40)
      acd257(76)=acd257(31)*acd257(76)
      acd257(74)=acd257(76)+acd257(85)+acd257(87)+acd257(74)
      acd257(74)=acd257(74)*acd257(81)
      acd257(76)=acd257(22)*acd257(64)
      acd257(81)=-acd257(20)*acd257(63)
      acd257(83)=acd257(32)*acd257(83)
      acd257(76)=acd257(83)+acd257(81)+acd257(76)
      acd257(76)=acd257(24)*acd257(76)
      acd257(81)=-acd257(31)*acd257(84)
      acd257(83)=-acd257(24)*acd257(54)
      acd257(81)=acd257(83)+acd257(81)
      acd257(81)=acd257(1)*acd257(81)
      acd257(83)=-acd257(31)*acd257(86)
      acd257(79)=-acd257(32)*acd257(79)
      acd257(84)=acd257(17)*acd257(73)
      acd257(79)=acd257(84)+acd257(79)
      acd257(79)=acd257(27)*acd257(79)
      acd257(77)=acd257(32)*acd257(77)
      acd257(84)=acd257(11)*acd257(17)*acd257(35)
      acd257(74)=acd257(82)+acd257(78)+acd257(81)+acd257(74)+acd257(77)+acd257(&
      &84)+acd257(76)+acd257(79)+acd257(83)
      brack(ninjaidxt1mu0)=acd257(80)
      brack(ninjaidxt0mu0)=acd257(74)
      brack(ninjaidxt0mu2)=acd257(75)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d257h4_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd257h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k4
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d257h4l131
