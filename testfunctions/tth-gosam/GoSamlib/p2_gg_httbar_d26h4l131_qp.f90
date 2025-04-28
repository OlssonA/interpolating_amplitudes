module     p2_gg_httbar_d26h4l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d26h4l131_qp.f90
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
      use p2_gg_httbar_abbrevd26h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd26
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd26h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(146) :: acd26
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd26(1)=abb26(17)
      acd26(2)=dotproduct(k2,ninjaE3)
      acd26(3)=abb26(14)
      acd26(4)=dotproduct(ninjaA,ninjaE3)
      acd26(5)=dotproduct(ninjaE3,spvae2k2)
      acd26(6)=abb26(11)
      acd26(7)=dotproduct(ninjaE3,spvak2e1)
      acd26(8)=abb26(12)
      acd26(9)=dotproduct(ninjaE3,spvak2l3)
      acd26(10)=abb26(15)
      acd26(11)=dotproduct(ninjaE3,spval5k2)
      acd26(12)=abb26(16)
      acd26(13)=dotproduct(ninjaE3,spvak1l3)
      acd26(14)=abb26(18)
      acd26(15)=dotproduct(ninjaE3,spvak2e2)
      acd26(16)=abb26(19)
      acd26(17)=dotproduct(ninjaE3,spval3k2)
      acd26(18)=abb26(20)
      acd26(19)=dotproduct(ninjaE3,spval4l3)
      acd26(20)=abb26(21)
      acd26(21)=dotproduct(ninjaE3,spvae2e1)
      acd26(22)=abb26(22)
      acd26(23)=dotproduct(ninjaE3,spvae1l5)
      acd26(24)=abb26(23)
      acd26(25)=dotproduct(ninjaE3,spval4k2)
      acd26(26)=abb26(24)
      acd26(27)=dotproduct(ninjaE3,spvak2k1)
      acd26(28)=abb26(25)
      acd26(29)=dotproduct(ninjaE3,spval5e1)
      acd26(30)=abb26(26)
      acd26(31)=dotproduct(ninjaE3,spval3k1)
      acd26(32)=abb26(27)
      acd26(33)=dotproduct(ninjaE3,spvak2l4)
      acd26(34)=abb26(28)
      acd26(35)=dotproduct(ninjaE3,spval5k1)
      acd26(36)=abb26(29)
      acd26(37)=dotproduct(ninjaE3,spvae2k1)
      acd26(38)=abb26(30)
      acd26(39)=dotproduct(ninjaE3,spvae2l4)
      acd26(40)=abb26(31)
      acd26(41)=dotproduct(ninjaE3,spvak1e2)
      acd26(42)=abb26(32)
      acd26(43)=dotproduct(ninjaE3,spvak1l5)
      acd26(44)=abb26(33)
      acd26(45)=dotproduct(ninjaE3,spval4l5)
      acd26(46)=abb26(34)
      acd26(47)=dotproduct(ninjaE3,spval4e2)
      acd26(48)=abb26(35)
      acd26(49)=dotproduct(ninjaE3,spvak2l5)
      acd26(50)=abb26(36)
      acd26(51)=dotproduct(ninjaE3,spvak1k2)
      acd26(52)=abb26(37)
      acd26(53)=dotproduct(ninjaE3,spvae2l3)
      acd26(54)=abb26(39)
      acd26(55)=dotproduct(ninjaE3,spval5l4)
      acd26(56)=abb26(40)
      acd26(57)=dotproduct(ninjaE3,spval3e2)
      acd26(58)=abb26(42)
      acd26(59)=dotproduct(ninjaE3,spval5l3)
      acd26(60)=abb26(43)
      acd26(61)=dotproduct(ninjaE3,spval3l5)
      acd26(62)=abb26(44)
      acd26(63)=dotproduct(ninjaE3,spval3l4)
      acd26(64)=abb26(45)
      acd26(65)=dotproduct(ninjaE3,spvae1e2)
      acd26(66)=abb26(52)
      acd26(67)=dotproduct(ninjaE3,spvae1l3)
      acd26(68)=abb26(57)
      acd26(69)=dotproduct(ninjaE3,spvae1k2)
      acd26(70)=abb26(72)
      acd26(71)=dotproduct(ninjaE3,spval3e1)
      acd26(72)=abb26(73)
      acd26(73)=dotproduct(k2,ninjaA)
      acd26(74)=dotproduct(ninjaA,ninjaA)
      acd26(75)=dotproduct(ninjaA,spvae2k2)
      acd26(76)=dotproduct(ninjaA,spvak2e1)
      acd26(77)=dotproduct(ninjaA,spvak2l3)
      acd26(78)=dotproduct(ninjaA,spval5k2)
      acd26(79)=dotproduct(ninjaA,spvak1l3)
      acd26(80)=dotproduct(ninjaA,spvak2e2)
      acd26(81)=dotproduct(ninjaA,spval3k2)
      acd26(82)=dotproduct(ninjaA,spval4l3)
      acd26(83)=dotproduct(ninjaA,spvae2e1)
      acd26(84)=dotproduct(ninjaA,spvae1l5)
      acd26(85)=dotproduct(ninjaA,spval4k2)
      acd26(86)=dotproduct(ninjaA,spvak2k1)
      acd26(87)=dotproduct(ninjaA,spval5e1)
      acd26(88)=dotproduct(ninjaA,spval3k1)
      acd26(89)=dotproduct(ninjaA,spvak2l4)
      acd26(90)=dotproduct(ninjaA,spval5k1)
      acd26(91)=dotproduct(ninjaA,spvae2k1)
      acd26(92)=dotproduct(ninjaA,spvae2l4)
      acd26(93)=dotproduct(ninjaA,spvak1e2)
      acd26(94)=dotproduct(ninjaA,spvak1l5)
      acd26(95)=dotproduct(ninjaA,spval4l5)
      acd26(96)=dotproduct(ninjaA,spval4e2)
      acd26(97)=dotproduct(ninjaA,spvak2l5)
      acd26(98)=dotproduct(ninjaA,spvak1k2)
      acd26(99)=dotproduct(ninjaA,spvae2l3)
      acd26(100)=dotproduct(ninjaA,spval5l4)
      acd26(101)=dotproduct(ninjaA,spval3e2)
      acd26(102)=dotproduct(ninjaA,spval5l3)
      acd26(103)=dotproduct(ninjaA,spval3l5)
      acd26(104)=dotproduct(ninjaA,spval3l4)
      acd26(105)=dotproduct(ninjaA,spvae1e2)
      acd26(106)=dotproduct(ninjaA,spvae1l3)
      acd26(107)=dotproduct(ninjaA,spvae1k2)
      acd26(108)=dotproduct(ninjaA,spval3e1)
      acd26(109)=abb26(13)
      acd26(110)=acd26(2)*acd26(3)
      acd26(111)=acd26(4)*acd26(1)
      acd26(112)=acd26(5)*acd26(6)
      acd26(113)=acd26(7)*acd26(8)
      acd26(114)=acd26(9)*acd26(10)
      acd26(115)=acd26(11)*acd26(12)
      acd26(116)=acd26(13)*acd26(14)
      acd26(117)=acd26(15)*acd26(16)
      acd26(118)=acd26(17)*acd26(18)
      acd26(119)=acd26(19)*acd26(20)
      acd26(120)=acd26(21)*acd26(22)
      acd26(121)=acd26(23)*acd26(24)
      acd26(122)=acd26(25)*acd26(26)
      acd26(123)=acd26(27)*acd26(28)
      acd26(124)=acd26(29)*acd26(30)
      acd26(125)=acd26(31)*acd26(32)
      acd26(126)=acd26(33)*acd26(34)
      acd26(127)=acd26(35)*acd26(36)
      acd26(128)=acd26(37)*acd26(38)
      acd26(129)=acd26(39)*acd26(40)
      acd26(130)=acd26(41)*acd26(42)
      acd26(131)=acd26(43)*acd26(44)
      acd26(132)=acd26(45)*acd26(46)
      acd26(133)=acd26(47)*acd26(48)
      acd26(134)=acd26(49)*acd26(50)
      acd26(135)=acd26(51)*acd26(52)
      acd26(136)=acd26(53)*acd26(54)
      acd26(137)=acd26(55)*acd26(56)
      acd26(138)=acd26(57)*acd26(58)
      acd26(139)=acd26(59)*acd26(60)
      acd26(140)=acd26(61)*acd26(62)
      acd26(141)=acd26(63)*acd26(64)
      acd26(142)=-acd26(65)*acd26(66)
      acd26(143)=acd26(67)*acd26(68)
      acd26(144)=acd26(69)*acd26(70)
      acd26(145)=-acd26(71)*acd26(72)
      acd26(110)=acd26(145)+acd26(144)+acd26(143)+acd26(142)+acd26(141)+acd26(1&
      &40)+acd26(139)+acd26(138)+acd26(137)+acd26(136)+acd26(135)+acd26(134)+ac&
      &d26(133)+acd26(132)+acd26(131)+acd26(130)+acd26(129)+acd26(128)+acd26(12&
      &7)+acd26(126)+acd26(125)+acd26(124)+acd26(123)+acd26(122)+acd26(121)+acd&
      &26(120)+acd26(119)+acd26(118)+acd26(117)+acd26(116)+acd26(115)+acd26(114&
      &)+acd26(113)+acd26(112)+acd26(110)+2.0_ki*acd26(111)
      acd26(111)=ninjaP+acd26(74)
      acd26(111)=acd26(1)*acd26(111)
      acd26(112)=acd26(73)*acd26(3)
      acd26(113)=acd26(75)*acd26(6)
      acd26(114)=acd26(76)*acd26(8)
      acd26(115)=acd26(77)*acd26(10)
      acd26(116)=acd26(78)*acd26(12)
      acd26(117)=acd26(79)*acd26(14)
      acd26(118)=acd26(80)*acd26(16)
      acd26(119)=acd26(81)*acd26(18)
      acd26(120)=acd26(82)*acd26(20)
      acd26(121)=acd26(83)*acd26(22)
      acd26(122)=acd26(84)*acd26(24)
      acd26(123)=acd26(85)*acd26(26)
      acd26(124)=acd26(86)*acd26(28)
      acd26(125)=acd26(87)*acd26(30)
      acd26(126)=acd26(88)*acd26(32)
      acd26(127)=acd26(89)*acd26(34)
      acd26(128)=acd26(90)*acd26(36)
      acd26(129)=acd26(91)*acd26(38)
      acd26(130)=acd26(92)*acd26(40)
      acd26(131)=acd26(93)*acd26(42)
      acd26(132)=acd26(94)*acd26(44)
      acd26(133)=acd26(95)*acd26(46)
      acd26(134)=acd26(96)*acd26(48)
      acd26(135)=acd26(97)*acd26(50)
      acd26(136)=acd26(98)*acd26(52)
      acd26(137)=acd26(99)*acd26(54)
      acd26(138)=acd26(100)*acd26(56)
      acd26(139)=acd26(101)*acd26(58)
      acd26(140)=acd26(102)*acd26(60)
      acd26(141)=acd26(103)*acd26(62)
      acd26(142)=acd26(104)*acd26(64)
      acd26(143)=-acd26(105)*acd26(66)
      acd26(144)=acd26(106)*acd26(68)
      acd26(145)=acd26(107)*acd26(70)
      acd26(146)=-acd26(108)*acd26(72)
      acd26(111)=acd26(109)+acd26(146)+acd26(145)+acd26(144)+acd26(143)+acd26(1&
      &42)+acd26(141)+acd26(140)+acd26(139)+acd26(138)+acd26(137)+acd26(136)+ac&
      &d26(135)+acd26(134)+acd26(133)+acd26(132)+acd26(131)+acd26(130)+acd26(12&
      &9)+acd26(128)+acd26(127)+acd26(126)+acd26(125)+acd26(124)+acd26(123)+acd&
      &26(122)+acd26(121)+acd26(120)+acd26(119)+acd26(118)+acd26(117)+acd26(116&
      &)+acd26(115)+acd26(114)+acd26(113)+acd26(112)+acd26(111)
      brack(ninjaidxt1mu0)=acd26(110)
      brack(ninjaidxt0mu0)=acd26(111)
      brack(ninjaidxt0mu2)=acd26(1)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d26h4_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd26h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k5
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
end module     p2_gg_httbar_d26h4l131_qp
