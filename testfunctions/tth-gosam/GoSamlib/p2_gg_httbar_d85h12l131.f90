module     p2_gg_httbar_d85h12l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d85h12l131.f90
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
      use p2_gg_httbar_abbrevd85h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(13) :: acd85
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd85(1)=dotproduct(e2,ninjaE3)
      acd85(2)=dotproduct(ninjaE3,spvak2e1)
      acd85(3)=dotproduct(ninjaE3,spvae1l5)
      acd85(4)=abb85(14)
      acd85(5)=dotproduct(ninjaE3,spvae1l4)
      acd85(6)=abb85(22)
      acd85(7)=dotproduct(ninjaE3,spvae1l3)
      acd85(8)=abb85(26)
      acd85(9)=dotproduct(ninjaE3,spval3e1)
      acd85(10)=abb85(76)
      acd85(11)=acd85(4)*acd85(3)
      acd85(12)=acd85(6)*acd85(5)
      acd85(13)=acd85(8)*acd85(7)
      acd85(11)=acd85(13)+acd85(11)+acd85(12)
      acd85(11)=acd85(11)*acd85(2)
      acd85(12)=acd85(10)*acd85(9)*acd85(3)
      acd85(11)=acd85(12)+acd85(11)
      acd85(11)=acd85(1)*acd85(11)
      brack(ninjaidxt2mu0)=acd85(11)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd85h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(97) :: acd85
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd85(1)=dotproduct(e2,ninjaE3)
      acd85(2)=dotproduct(ninjaE3,spvae1l5)
      acd85(3)=dotproduct(ninjaE4,spvak2e1)
      acd85(4)=abb85(14)
      acd85(5)=dotproduct(ninjaE4,spval3e1)
      acd85(6)=abb85(76)
      acd85(7)=dotproduct(ninjaE3,spvak2e1)
      acd85(8)=dotproduct(ninjaE4,spvae1l5)
      acd85(9)=dotproduct(ninjaE4,spvae1l4)
      acd85(10)=abb85(22)
      acd85(11)=dotproduct(ninjaE4,spvae1l3)
      acd85(12)=abb85(26)
      acd85(13)=dotproduct(ninjaE3,spvae1l4)
      acd85(14)=dotproduct(ninjaE3,spvae1l3)
      acd85(15)=dotproduct(ninjaE3,spval3e1)
      acd85(16)=dotproduct(e2,ninjaE4)
      acd85(17)=dotproduct(ninjaE3,spvae1e2)
      acd85(18)=abb85(57)
      acd85(19)=dotproduct(ninjaE3,spvae2e1)
      acd85(20)=abb85(49)
      acd85(21)=dotproduct(e2,ninjaA)
      acd85(22)=dotproduct(ninjaA,spvae1l5)
      acd85(23)=dotproduct(ninjaA,spvak2e1)
      acd85(24)=dotproduct(ninjaA,spvae1l4)
      acd85(25)=dotproduct(ninjaA,spvae1l3)
      acd85(26)=dotproduct(ninjaA,spval3e1)
      acd85(27)=abb85(16)
      acd85(28)=abb85(18)
      acd85(29)=abb85(24)
      acd85(30)=abb85(47)
      acd85(31)=abb85(50)
      acd85(32)=dotproduct(ninjaA,ninjaE3)
      acd85(33)=abb85(11)
      acd85(34)=dotproduct(ninjaE3,spvak1l4)
      acd85(35)=abb85(32)
      acd85(36)=dotproduct(ninjaE3,spvak1l3)
      acd85(37)=abb85(35)
      acd85(38)=abb85(37)
      acd85(39)=dotproduct(ninjaE3,spval5e1)
      acd85(40)=abb85(65)
      acd85(41)=abb85(9)
      acd85(42)=abb85(12)
      acd85(43)=abb85(74)
      acd85(44)=abb85(23)
      acd85(45)=dotproduct(ninjaE3,spvae1k2)
      acd85(46)=abb85(44)
      acd85(47)=abb85(55)
      acd85(48)=dotproduct(ninjaE3,spval3k1)
      acd85(49)=abb85(29)
      acd85(50)=dotproduct(ninjaE3,spvak2k1)
      acd85(51)=abb85(31)
      acd85(52)=abb85(15)
      acd85(53)=abb85(25)
      acd85(54)=abb85(20)
      acd85(55)=dotproduct(ninjaA,ninjaA)
      acd85(56)=dotproduct(ninjaA,spvae1e2)
      acd85(57)=dotproduct(ninjaA,spvae2e1)
      acd85(58)=abb85(21)
      acd85(59)=dotproduct(ninjaA,spvae1k2)
      acd85(60)=dotproduct(ninjaA,spvak1l4)
      acd85(61)=dotproduct(ninjaA,spval3k1)
      acd85(62)=dotproduct(ninjaA,spvak2k1)
      acd85(63)=dotproduct(ninjaA,spvak1l3)
      acd85(64)=dotproduct(ninjaA,spval5e1)
      acd85(65)=abb85(8)
      acd85(66)=abb85(10)
      acd85(67)=abb85(33)
      acd85(68)=abb85(13)
      acd85(69)=abb85(67)
      acd85(70)=abb85(17)
      acd85(71)=abb85(42)
      acd85(72)=abb85(27)
      acd85(73)=abb85(28)
      acd85(74)=abb85(30)
      acd85(75)=abb85(34)
      acd85(76)=abb85(73)
      acd85(77)=abb85(61)
      acd85(78)=acd85(12)*acd85(11)
      acd85(79)=acd85(10)*acd85(9)
      acd85(80)=acd85(4)*acd85(8)
      acd85(78)=acd85(80)+acd85(78)+acd85(79)
      acd85(78)=acd85(78)*acd85(7)
      acd85(79)=acd85(6)*acd85(5)
      acd85(80)=acd85(4)*acd85(3)
      acd85(79)=acd85(79)+acd85(80)
      acd85(79)=acd85(79)*acd85(2)
      acd85(80)=acd85(14)*acd85(12)
      acd85(81)=acd85(13)*acd85(10)
      acd85(80)=acd85(80)+acd85(81)
      acd85(81)=acd85(3)*acd85(80)
      acd85(82)=acd85(15)*acd85(6)
      acd85(83)=acd85(82)*acd85(8)
      acd85(78)=acd85(78)+acd85(79)+acd85(81)+acd85(83)
      acd85(79)=acd85(1)*acd85(78)
      acd85(81)=acd85(2)*acd85(16)
      acd85(83)=acd85(82)*acd85(81)
      acd85(84)=acd85(16)*acd85(80)
      acd85(81)=acd85(4)*acd85(81)
      acd85(81)=acd85(81)+acd85(84)
      acd85(81)=acd85(7)*acd85(81)
      acd85(84)=acd85(17)*acd85(18)
      acd85(85)=acd85(19)*acd85(20)
      acd85(79)=acd85(79)+acd85(81)+acd85(83)+acd85(84)+acd85(85)
      acd85(81)=acd85(21)*acd85(80)
      acd85(83)=acd85(14)*acd85(53)
      acd85(84)=-acd85(13)*acd85(52)
      acd85(85)=acd85(17)*acd85(33)
      acd85(86)=acd85(21)*acd85(4)
      acd85(86)=acd85(42)+acd85(86)
      acd85(86)=acd85(2)*acd85(86)
      acd85(81)=acd85(86)+acd85(81)+acd85(85)+acd85(83)+acd85(84)
      acd85(81)=acd85(7)*acd85(81)
      acd85(83)=acd85(39)*acd85(40)
      acd85(84)=acd85(36)*acd85(37)
      acd85(85)=acd85(34)*acd85(35)
      acd85(86)=2.0_ki*acd85(32)
      acd85(87)=acd85(86)*acd85(18)
      acd85(83)=acd85(83)+acd85(84)+acd85(85)+acd85(87)
      acd85(84)=acd85(15)*acd85(38)
      acd85(84)=acd85(84)+acd85(83)
      acd85(84)=acd85(17)*acd85(84)
      acd85(85)=acd85(50)*acd85(51)
      acd85(87)=acd85(48)*acd85(49)
      acd85(88)=acd85(45)*acd85(46)
      acd85(89)=acd85(86)*acd85(20)
      acd85(85)=acd85(85)+acd85(87)+acd85(88)+acd85(89)
      acd85(87)=acd85(14)*acd85(47)
      acd85(88)=acd85(13)*acd85(44)
      acd85(87)=acd85(88)+acd85(87)+acd85(85)
      acd85(87)=acd85(19)*acd85(87)
      acd85(88)=acd85(23)*acd85(12)
      acd85(88)=acd85(88)+acd85(30)
      acd85(88)=acd85(88)*acd85(14)
      acd85(89)=acd85(23)*acd85(10)
      acd85(89)=acd85(89)+acd85(29)
      acd85(89)=acd85(89)*acd85(13)
      acd85(90)=acd85(6)*acd85(22)
      acd85(90)=acd85(90)+acd85(31)
      acd85(91)=acd85(90)*acd85(15)
      acd85(88)=acd85(91)+acd85(88)+acd85(89)
      acd85(89)=acd85(12)*acd85(25)
      acd85(91)=acd85(10)*acd85(24)
      acd85(92)=acd85(4)*acd85(22)
      acd85(89)=acd85(28)+acd85(89)+acd85(91)+acd85(92)
      acd85(91)=acd85(7)*acd85(89)
      acd85(92)=acd85(23)*acd85(4)
      acd85(93)=acd85(6)*acd85(26)
      acd85(92)=acd85(27)+acd85(92)+acd85(93)
      acd85(93)=acd85(2)*acd85(92)
      acd85(91)=acd85(91)+acd85(93)+acd85(88)
      acd85(91)=acd85(1)*acd85(91)
      acd85(93)=acd85(15)*acd85(43)
      acd85(94)=acd85(21)*acd85(82)
      acd85(95)=acd85(19)*acd85(41)
      acd85(93)=acd85(95)+acd85(93)+acd85(94)
      acd85(93)=acd85(2)*acd85(93)
      acd85(81)=acd85(91)+acd85(81)+acd85(93)+acd85(84)+acd85(87)
      acd85(84)=ninjaP+acd85(55)
      acd85(87)=acd85(20)*acd85(84)
      acd85(91)=acd85(51)*acd85(62)
      acd85(93)=acd85(49)*acd85(61)
      acd85(94)=acd85(46)*acd85(59)
      acd85(95)=acd85(25)*acd85(47)
      acd85(96)=acd85(24)*acd85(44)
      acd85(97)=acd85(22)*acd85(41)
      acd85(87)=acd85(97)+acd85(96)+acd85(95)+acd85(94)+acd85(93)+acd85(67)+acd&
      &85(91)+acd85(87)
      acd85(87)=acd85(19)*acd85(87)
      acd85(84)=acd85(18)*acd85(84)
      acd85(91)=acd85(40)*acd85(64)
      acd85(93)=acd85(37)*acd85(63)
      acd85(94)=acd85(35)*acd85(60)
      acd85(95)=acd85(26)*acd85(38)
      acd85(96)=acd85(23)*acd85(33)
      acd85(84)=acd85(96)+acd85(95)+acd85(94)+acd85(93)+acd85(65)+acd85(91)+acd&
      &85(84)
      acd85(84)=acd85(17)*acd85(84)
      acd85(91)=acd85(21)*acd85(89)
      acd85(93)=acd85(2)*acd85(4)
      acd85(80)=acd85(93)+acd85(80)
      acd85(93)=ninjaP*acd85(16)
      acd85(80)=acd85(93)*acd85(80)
      acd85(94)=acd85(56)*acd85(33)
      acd85(95)=acd85(25)*acd85(53)
      acd85(96)=-acd85(24)*acd85(52)
      acd85(97)=acd85(22)*acd85(42)
      acd85(80)=acd85(91)+acd85(97)+acd85(96)+acd85(95)+acd85(68)+acd85(94)+acd&
      &85(80)
      acd85(80)=acd85(7)*acd85(80)
      acd85(78)=ninjaP*acd85(78)
      acd85(89)=acd85(23)*acd85(89)
      acd85(90)=acd85(26)*acd85(90)
      acd85(91)=acd85(25)*acd85(30)
      acd85(94)=acd85(24)*acd85(29)
      acd85(95)=acd85(22)*acd85(27)
      acd85(78)=acd85(89)+acd85(95)+acd85(94)+acd85(91)+acd85(54)+acd85(78)+acd&
      &85(90)
      acd85(78)=acd85(1)*acd85(78)
      acd85(82)=acd85(82)*acd85(93)
      acd85(89)=acd85(21)*acd85(92)
      acd85(90)=acd85(26)*acd85(43)
      acd85(91)=acd85(57)*acd85(41)
      acd85(92)=acd85(23)*acd85(42)
      acd85(82)=acd85(89)+acd85(82)+acd85(92)+acd85(91)+acd85(66)+acd85(90)
      acd85(82)=acd85(2)*acd85(82)
      acd85(83)=acd85(56)*acd85(83)
      acd85(85)=acd85(57)*acd85(85)
      acd85(88)=acd85(21)*acd85(88)
      acd85(89)=acd85(56)*acd85(38)
      acd85(90)=acd85(22)*acd85(43)
      acd85(89)=acd85(90)+acd85(76)+acd85(89)
      acd85(89)=acd85(15)*acd85(89)
      acd85(90)=acd85(57)*acd85(47)
      acd85(91)=acd85(23)*acd85(53)
      acd85(90)=acd85(91)+acd85(71)+acd85(90)
      acd85(90)=acd85(14)*acd85(90)
      acd85(91)=acd85(57)*acd85(44)
      acd85(92)=-acd85(23)*acd85(52)
      acd85(91)=acd85(92)+acd85(69)+acd85(91)
      acd85(91)=acd85(13)*acd85(91)
      acd85(92)=acd85(50)*acd85(74)
      acd85(93)=acd85(48)*acd85(73)
      acd85(94)=acd85(45)*acd85(70)
      acd85(95)=acd85(39)*acd85(77)
      acd85(96)=acd85(36)*acd85(75)
      acd85(97)=acd85(34)*acd85(72)
      acd85(86)=acd85(58)*acd85(86)
      acd85(78)=acd85(78)+acd85(80)+acd85(82)+acd85(87)+acd85(88)+acd85(84)+acd&
      &85(91)+acd85(90)+acd85(89)+acd85(85)+acd85(83)+acd85(86)+acd85(97)+acd85&
      &(96)+acd85(95)+acd85(94)+acd85(92)+acd85(93)
      brack(ninjaidxt1mu0)=acd85(81)
      brack(ninjaidxt0mu0)=acd85(78)
      brack(ninjaidxt0mu2)=acd85(79)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d85h12_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd85h12
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k5-k2
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
end module     p2_gg_httbar_d85h12l131
