module     p2_gg_httbar_d91h8l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d91h8l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd91h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(14) :: acd91
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd91(1)=dotproduct(e2,ninjaE3)
      acd91(2)=dotproduct(ninjaE3,spvae1k2)
      acd91(3)=dotproduct(ninjaE3,spval3e1)
      acd91(4)=abb91(17)
      acd91(5)=dotproduct(ninjaE3,spvak2e1)
      acd91(6)=abb91(20)
      acd91(7)=dotproduct(ninjaE3,spval4e1)
      acd91(8)=dotproduct(ninjaE3,spvae1l5)
      acd91(9)=abb91(30)
      acd91(10)=dotproduct(ninjaE3,spvae1l3)
      acd91(11)=abb91(57)
      acd91(12)=acd91(9)*acd91(8)
      acd91(13)=acd91(11)*acd91(10)
      acd91(12)=acd91(13)+acd91(12)
      acd91(12)=acd91(12)*acd91(7)
      acd91(13)=acd91(4)*acd91(3)
      acd91(14)=acd91(6)*acd91(5)
      acd91(13)=acd91(13)+acd91(14)
      acd91(13)=acd91(13)*acd91(2)
      acd91(12)=acd91(13)+acd91(12)
      acd91(12)=acd91(1)*acd91(12)
      brack(ninjaidxt2mu0)=acd91(12)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd91h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(100) :: acd91
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd91(1)=dotproduct(e2,ninjaE3)
      acd91(2)=dotproduct(ninjaE3,spval4e1)
      acd91(3)=dotproduct(ninjaE4,spvae1l3)
      acd91(4)=abb91(57)
      acd91(5)=dotproduct(ninjaE4,spvae1l5)
      acd91(6)=abb91(30)
      acd91(7)=dotproduct(ninjaE3,spvae1k2)
      acd91(8)=dotproduct(ninjaE4,spval3e1)
      acd91(9)=abb91(17)
      acd91(10)=dotproduct(ninjaE4,spvak2e1)
      acd91(11)=abb91(20)
      acd91(12)=dotproduct(ninjaE3,spval3e1)
      acd91(13)=dotproduct(ninjaE4,spvae1k2)
      acd91(14)=dotproduct(ninjaE3,spvak2e1)
      acd91(15)=dotproduct(ninjaE3,spvae1l3)
      acd91(16)=dotproduct(ninjaE4,spval4e1)
      acd91(17)=dotproduct(ninjaE3,spvae1l5)
      acd91(18)=dotproduct(e2,ninjaE4)
      acd91(19)=dotproduct(ninjaE3,spvae2e1)
      acd91(20)=abb91(23)
      acd91(21)=dotproduct(ninjaE3,spvae1e2)
      acd91(22)=abb91(72)
      acd91(23)=dotproduct(e2,ninjaA)
      acd91(24)=dotproduct(ninjaA,spval4e1)
      acd91(25)=dotproduct(ninjaA,spvae1k2)
      acd91(26)=dotproduct(ninjaA,spval3e1)
      acd91(27)=dotproduct(ninjaA,spvak2e1)
      acd91(28)=dotproduct(ninjaA,spvae1l3)
      acd91(29)=dotproduct(ninjaA,spvae1l5)
      acd91(30)=abb91(42)
      acd91(31)=abb91(16)
      acd91(32)=abb91(37)
      acd91(33)=abb91(15)
      acd91(34)=abb91(55)
      acd91(35)=abb91(50)
      acd91(36)=dotproduct(ninjaA,ninjaE3)
      acd91(37)=abb91(10)
      acd91(38)=dotproduct(ninjaE3,spval3k1)
      acd91(39)=abb91(24)
      acd91(40)=abb91(61)
      acd91(41)=dotproduct(ninjaE3,spvak2k1)
      acd91(42)=abb91(27)
      acd91(43)=abb91(81)
      acd91(44)=dotproduct(ninjaE3,spvae1l4)
      acd91(45)=abb91(43)
      acd91(46)=abb91(25)
      acd91(47)=abb91(64)
      acd91(48)=abb91(45)
      acd91(49)=abb91(11)
      acd91(50)=abb91(19)
      acd91(51)=abb91(82)
      acd91(52)=abb91(18)
      acd91(53)=dotproduct(ninjaE3,spvak1l5)
      acd91(54)=abb91(33)
      acd91(55)=dotproduct(ninjaE3,spvak1l3)
      acd91(56)=abb91(35)
      acd91(57)=abb91(32)
      acd91(58)=dotproduct(ninjaA,ninjaA)
      acd91(59)=dotproduct(ninjaA,spvae2e1)
      acd91(60)=dotproduct(ninjaA,spvae1e2)
      acd91(61)=abb91(29)
      acd91(62)=dotproduct(ninjaA,spval3k1)
      acd91(63)=dotproduct(ninjaA,spvak2k1)
      acd91(64)=dotproduct(ninjaA,spvak1l5)
      acd91(65)=dotproduct(ninjaA,spvak1l3)
      acd91(66)=dotproduct(ninjaA,spvae1l4)
      acd91(67)=abb91(8)
      acd91(68)=abb91(9)
      acd91(69)=abb91(13)
      acd91(70)=abb91(68)
      acd91(71)=abb91(12)
      acd91(72)=abb91(63)
      acd91(73)=abb91(21)
      acd91(74)=abb91(22)
      acd91(75)=abb91(26)
      acd91(76)=abb91(28)
      acd91(77)=abb91(31)
      acd91(78)=abb91(34)
      acd91(79)=abb91(69)
      acd91(80)=acd91(11)*acd91(10)
      acd91(81)=acd91(9)*acd91(8)
      acd91(80)=acd91(80)+acd91(81)
      acd91(80)=acd91(80)*acd91(7)
      acd91(81)=acd91(6)*acd91(5)
      acd91(82)=acd91(4)*acd91(3)
      acd91(81)=acd91(81)+acd91(82)
      acd91(81)=acd91(81)*acd91(2)
      acd91(82)=acd91(17)*acd91(6)
      acd91(83)=acd91(15)*acd91(4)
      acd91(82)=acd91(82)+acd91(83)
      acd91(83)=acd91(16)*acd91(82)
      acd91(84)=acd91(14)*acd91(11)
      acd91(85)=acd91(12)*acd91(9)
      acd91(84)=acd91(84)+acd91(85)
      acd91(85)=acd91(13)*acd91(84)
      acd91(80)=acd91(81)+acd91(80)+acd91(83)+acd91(85)
      acd91(81)=acd91(1)*acd91(80)
      acd91(83)=acd91(7)*acd91(84)
      acd91(85)=acd91(2)*acd91(82)
      acd91(83)=acd91(85)+acd91(83)
      acd91(83)=acd91(18)*acd91(83)
      acd91(85)=acd91(21)*acd91(22)
      acd91(86)=acd91(19)*acd91(20)
      acd91(81)=acd91(81)+acd91(85)+acd91(86)+acd91(83)
      acd91(83)=acd91(44)*acd91(45)
      acd91(85)=acd91(41)*acd91(42)
      acd91(86)=acd91(38)*acd91(39)
      acd91(87)=2.0_ki*acd91(36)
      acd91(88)=acd91(87)*acd91(20)
      acd91(83)=acd91(83)+acd91(85)+acd91(86)+acd91(88)
      acd91(85)=acd91(17)*acd91(43)
      acd91(86)=acd91(15)*acd91(40)
      acd91(85)=acd91(86)+acd91(85)+acd91(83)
      acd91(85)=acd91(19)*acd91(85)
      acd91(86)=acd91(23)*acd91(84)
      acd91(88)=acd91(14)*acd91(50)
      acd91(89)=acd91(12)*acd91(49)
      acd91(90)=acd91(19)*acd91(37)
      acd91(86)=acd91(86)+acd91(90)+acd91(88)+acd91(89)
      acd91(86)=acd91(7)*acd91(86)
      acd91(88)=acd91(23)*acd91(82)
      acd91(89)=acd91(17)*acd91(48)
      acd91(90)=acd91(15)*acd91(47)
      acd91(91)=acd91(21)*acd91(46)
      acd91(88)=acd91(88)+acd91(91)+acd91(89)+acd91(90)
      acd91(88)=acd91(2)*acd91(88)
      acd91(89)=acd91(9)*acd91(25)
      acd91(89)=acd91(89)+acd91(32)
      acd91(89)=acd91(89)*acd91(12)
      acd91(90)=acd91(11)*acd91(25)
      acd91(90)=acd91(90)+acd91(33)
      acd91(90)=acd91(90)*acd91(14)
      acd91(91)=acd91(6)*acd91(24)
      acd91(91)=acd91(91)+acd91(35)
      acd91(92)=acd91(91)*acd91(17)
      acd91(93)=acd91(4)*acd91(24)
      acd91(93)=acd91(93)+acd91(34)
      acd91(94)=acd91(93)*acd91(15)
      acd91(89)=acd91(89)+acd91(90)+acd91(92)+acd91(94)
      acd91(90)=acd91(11)*acd91(27)
      acd91(92)=acd91(9)*acd91(26)
      acd91(90)=acd91(31)+acd91(90)+acd91(92)
      acd91(92)=acd91(7)*acd91(90)
      acd91(94)=acd91(6)*acd91(29)
      acd91(95)=acd91(4)*acd91(28)
      acd91(94)=acd91(30)+acd91(94)+acd91(95)
      acd91(95)=acd91(2)*acd91(94)
      acd91(92)=acd91(95)+acd91(92)+acd91(89)
      acd91(92)=acd91(1)*acd91(92)
      acd91(95)=acd91(55)*acd91(56)
      acd91(96)=acd91(53)*acd91(54)
      acd91(97)=acd91(87)*acd91(22)
      acd91(95)=acd91(97)+acd91(95)+acd91(96)
      acd91(96)=acd91(14)*acd91(52)
      acd91(97)=acd91(12)*acd91(51)
      acd91(96)=acd91(97)+acd91(96)+acd91(95)
      acd91(96)=acd91(21)*acd91(96)
      acd91(85)=acd91(92)+acd91(88)+acd91(86)+acd91(96)+acd91(85)
      acd91(86)=ninjaP+acd91(58)
      acd91(88)=acd91(20)*acd91(86)
      acd91(92)=acd91(45)*acd91(66)
      acd91(96)=acd91(42)*acd91(63)
      acd91(97)=acd91(39)*acd91(62)
      acd91(98)=acd91(29)*acd91(43)
      acd91(99)=acd91(28)*acd91(40)
      acd91(100)=acd91(25)*acd91(37)
      acd91(88)=acd91(100)+acd91(99)+acd91(98)+acd91(97)+acd91(96)+acd91(67)+ac&
      &d91(92)+acd91(88)
      acd91(88)=acd91(19)*acd91(88)
      acd91(80)=ninjaP*acd91(80)
      acd91(92)=acd91(25)*acd91(90)
      acd91(91)=acd91(29)*acd91(91)
      acd91(93)=acd91(28)*acd91(93)
      acd91(96)=acd91(27)*acd91(33)
      acd91(97)=acd91(26)*acd91(32)
      acd91(98)=acd91(24)*acd91(30)
      acd91(80)=acd91(98)+acd91(97)+acd91(96)+acd91(57)+acd91(80)+acd91(92)+acd&
      &91(93)+acd91(91)
      acd91(80)=acd91(1)*acd91(80)
      acd91(86)=acd91(22)*acd91(86)
      acd91(91)=acd91(56)*acd91(65)
      acd91(92)=acd91(54)*acd91(64)
      acd91(93)=acd91(27)*acd91(52)
      acd91(96)=acd91(26)*acd91(51)
      acd91(97)=acd91(24)*acd91(46)
      acd91(86)=acd91(97)+acd91(96)+acd91(93)+acd91(92)+acd91(72)+acd91(91)+acd&
      &91(86)
      acd91(86)=acd91(21)*acd91(86)
      acd91(91)=ninjaP*acd91(18)
      acd91(84)=acd91(91)*acd91(84)
      acd91(90)=acd91(23)*acd91(90)
      acd91(92)=acd91(27)*acd91(50)
      acd91(93)=acd91(26)*acd91(49)
      acd91(96)=acd91(59)*acd91(37)
      acd91(84)=acd91(90)+acd91(96)+acd91(93)+acd91(69)+acd91(92)+acd91(84)
      acd91(84)=acd91(7)*acd91(84)
      acd91(82)=acd91(91)*acd91(82)
      acd91(90)=acd91(23)*acd91(94)
      acd91(91)=acd91(60)*acd91(46)
      acd91(92)=acd91(29)*acd91(48)
      acd91(93)=acd91(28)*acd91(47)
      acd91(82)=acd91(90)+acd91(93)+acd91(92)+acd91(68)+acd91(91)+acd91(82)
      acd91(82)=acd91(2)*acd91(82)
      acd91(83)=acd91(59)*acd91(83)
      acd91(89)=acd91(23)*acd91(89)
      acd91(90)=acd91(60)*acd91(95)
      acd91(91)=acd91(59)*acd91(43)
      acd91(92)=acd91(24)*acd91(48)
      acd91(91)=acd91(92)+acd91(77)+acd91(91)
      acd91(91)=acd91(17)*acd91(91)
      acd91(92)=acd91(59)*acd91(40)
      acd91(93)=acd91(24)*acd91(47)
      acd91(92)=acd91(93)+acd91(74)+acd91(92)
      acd91(92)=acd91(15)*acd91(92)
      acd91(93)=acd91(60)*acd91(52)
      acd91(94)=acd91(25)*acd91(50)
      acd91(93)=acd91(94)+acd91(71)+acd91(93)
      acd91(93)=acd91(14)*acd91(93)
      acd91(94)=acd91(60)*acd91(51)
      acd91(95)=acd91(25)*acd91(49)
      acd91(94)=acd91(95)+acd91(70)+acd91(94)
      acd91(94)=acd91(12)*acd91(94)
      acd91(95)=acd91(55)*acd91(78)
      acd91(96)=acd91(53)*acd91(76)
      acd91(97)=acd91(44)*acd91(79)
      acd91(98)=acd91(41)*acd91(75)
      acd91(99)=acd91(38)*acd91(73)
      acd91(87)=acd91(61)*acd91(87)
      acd91(80)=acd91(80)+acd91(82)+acd91(84)+acd91(89)+acd91(88)+acd91(86)+acd&
      &91(94)+acd91(93)+acd91(92)+acd91(91)+acd91(83)+acd91(90)+acd91(87)+acd91&
      &(99)+acd91(98)+acd91(97)+acd91(95)+acd91(96)
      brack(ninjaidxt1mu0)=acd91(85)
      brack(ninjaidxt0mu0)=acd91(80)
      brack(ninjaidxt0mu2)=acd91(81)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d91h8_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd91h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k4
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
end module     p2_gg_httbar_d91h8l131_qp
