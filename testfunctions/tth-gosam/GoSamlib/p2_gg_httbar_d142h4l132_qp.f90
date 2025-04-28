module     p2_gg_httbar_d142h4l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d142h4l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd142h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(32) :: acd142
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd142(1)=dotproduct(ninjaE3,spvak2e2)
      acd142(2)=dotproduct(ninjaE3,spvae2k2)
      acd142(3)=abb142(12)
      acd142(4)=dotproduct(ninjaE3,spvae2e1)
      acd142(5)=abb142(21)
      acd142(6)=dotproduct(ninjaE3,spvae2k1)
      acd142(7)=abb142(22)
      acd142(8)=dotproduct(ninjaE3,spvae2l5)
      acd142(9)=abb142(23)
      acd142(10)=dotproduct(ninjaE3,spvae2l4)
      acd142(11)=abb142(25)
      acd142(12)=dotproduct(ninjaE3,spvak1e2)
      acd142(13)=abb142(14)
      acd142(14)=dotproduct(ninjaE3,spvae1e2)
      acd142(15)=abb142(38)
      acd142(16)=dotproduct(ninjaE3,spval5e2)
      acd142(17)=abb142(41)
      acd142(18)=dotproduct(ninjaE3,spval4e2)
      acd142(19)=abb142(43)
      acd142(20)=abb142(90)
      acd142(21)=abb142(24)
      acd142(22)=abb142(16)
      acd142(23)=abb142(89)
      acd142(24)=abb142(86)
      acd142(25)=abb142(19)
      acd142(26)=abb142(94)
      acd142(27)=acd142(3)*acd142(2)
      acd142(28)=acd142(5)*acd142(4)
      acd142(29)=acd142(7)*acd142(6)
      acd142(30)=acd142(9)*acd142(8)
      acd142(31)=acd142(11)*acd142(10)
      acd142(27)=acd142(31)+acd142(30)+acd142(29)+acd142(28)+acd142(27)
      acd142(27)=acd142(1)*acd142(27)
      acd142(28)=-acd142(8)*acd142(20)
      acd142(29)=acd142(17)*acd142(2)
      acd142(30)=acd142(24)*acd142(4)
      acd142(31)=acd142(25)*acd142(6)
      acd142(32)=acd142(26)*acd142(10)
      acd142(28)=acd142(32)+acd142(31)+acd142(30)+acd142(29)+acd142(28)
      acd142(28)=acd142(16)*acd142(28)
      acd142(29)=-acd142(20)*acd142(6)
      acd142(30)=acd142(13)*acd142(2)
      acd142(31)=acd142(21)*acd142(10)
      acd142(29)=acd142(31)+acd142(30)+acd142(29)
      acd142(29)=acd142(12)*acd142(29)
      acd142(30)=acd142(15)*acd142(2)
      acd142(31)=acd142(22)*acd142(4)
      acd142(32)=acd142(23)*acd142(10)
      acd142(30)=acd142(32)+acd142(31)+acd142(30)
      acd142(30)=acd142(14)*acd142(30)
      acd142(31)=-acd142(20)*acd142(10)
      acd142(32)=acd142(19)*acd142(2)
      acd142(31)=acd142(32)+acd142(31)
      acd142(31)=acd142(18)*acd142(31)
      acd142(27)=acd142(28)+acd142(27)+acd142(30)+acd142(29)+acd142(31)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd142(27)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd142h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(90) :: acd142
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd142(1)=dotproduct(ninjaA1,spvak2e2)
      acd142(2)=dotproduct(ninjaE3,spvae2k2)
      acd142(3)=abb142(12)
      acd142(4)=dotproduct(ninjaE3,spvae2k1)
      acd142(5)=abb142(22)
      acd142(6)=dotproduct(ninjaE3,spvae2e1)
      acd142(7)=abb142(21)
      acd142(8)=dotproduct(ninjaE3,spvae2l5)
      acd142(9)=abb142(23)
      acd142(10)=dotproduct(ninjaE3,spvae2l4)
      acd142(11)=abb142(25)
      acd142(12)=dotproduct(ninjaA1,spvae2k2)
      acd142(13)=dotproduct(ninjaE3,spvak2e2)
      acd142(14)=dotproduct(ninjaE3,spvak1e2)
      acd142(15)=abb142(14)
      acd142(16)=dotproduct(ninjaE3,spvae1e2)
      acd142(17)=abb142(38)
      acd142(18)=dotproduct(ninjaE3,spval5e2)
      acd142(19)=abb142(41)
      acd142(20)=dotproduct(ninjaE3,spval4e2)
      acd142(21)=abb142(43)
      acd142(22)=dotproduct(ninjaA1,spvae2k1)
      acd142(23)=abb142(90)
      acd142(24)=abb142(19)
      acd142(25)=dotproduct(ninjaA1,spvak1e2)
      acd142(26)=abb142(24)
      acd142(27)=dotproduct(ninjaA1,spvae1e2)
      acd142(28)=abb142(16)
      acd142(29)=abb142(89)
      acd142(30)=dotproduct(ninjaA1,spvae2e1)
      acd142(31)=abb142(86)
      acd142(32)=dotproduct(ninjaA1,spval5e2)
      acd142(33)=abb142(94)
      acd142(34)=dotproduct(ninjaA1,spvae2l5)
      acd142(35)=dotproduct(ninjaA1,spvae2l4)
      acd142(36)=dotproduct(ninjaA1,spval4e2)
      acd142(37)=dotproduct(ninjaA0,ninjaE3)
      acd142(38)=abb142(18)
      acd142(39)=dotproduct(ninjaA0,spvak2e2)
      acd142(40)=dotproduct(ninjaA0,spvae2k2)
      acd142(41)=dotproduct(ninjaA0,spvae2k1)
      acd142(42)=dotproduct(ninjaA0,spvak1e2)
      acd142(43)=dotproduct(ninjaA0,spvae1e2)
      acd142(44)=dotproduct(ninjaA0,spvae2e1)
      acd142(45)=dotproduct(ninjaA0,spval5e2)
      acd142(46)=dotproduct(ninjaA0,spvae2l5)
      acd142(47)=dotproduct(ninjaA0,spvae2l4)
      acd142(48)=dotproduct(ninjaA0,spval4e2)
      acd142(49)=abb142(20)
      acd142(50)=abb142(17)
      acd142(51)=abb142(13)
      acd142(52)=abb142(28)
      acd142(53)=dotproduct(ninjaE3,spval3e2)
      acd142(54)=abb142(15)
      acd142(55)=abb142(70)
      acd142(56)=abb142(53)
      acd142(57)=abb142(71)
      acd142(58)=abb142(104)
      acd142(59)=abb142(72)
      acd142(60)=dotproduct(ninjaE3,spvae2l3)
      acd142(61)=abb142(27)
      acd142(62)=abb142(78)
      acd142(63)=acd142(5)*acd142(4)
      acd142(64)=acd142(7)*acd142(6)
      acd142(65)=acd142(9)*acd142(8)
      acd142(63)=acd142(65)+acd142(63)+acd142(64)
      acd142(64)=acd142(1)*acd142(63)
      acd142(65)=acd142(15)*acd142(14)
      acd142(66)=acd142(17)*acd142(16)
      acd142(67)=acd142(21)*acd142(20)
      acd142(65)=acd142(67)+acd142(65)+acd142(66)
      acd142(66)=acd142(12)*acd142(65)
      acd142(67)=acd142(8)*acd142(23)
      acd142(68)=acd142(24)*acd142(4)
      acd142(69)=acd142(31)*acd142(6)
      acd142(67)=-acd142(69)+acd142(67)-acd142(68)
      acd142(68)=-acd142(32)*acd142(67)
      acd142(69)=acd142(20)*acd142(23)
      acd142(70)=acd142(26)*acd142(14)
      acd142(71)=acd142(29)*acd142(16)
      acd142(69)=-acd142(71)+acd142(69)-acd142(70)
      acd142(70)=-acd142(35)*acd142(69)
      acd142(71)=acd142(14)*acd142(23)
      acd142(72)=acd142(5)*acd142(13)
      acd142(73)=acd142(24)*acd142(18)
      acd142(71)=-acd142(73)+acd142(71)-acd142(72)
      acd142(72)=-acd142(22)*acd142(71)
      acd142(73)=acd142(4)*acd142(23)
      acd142(74)=acd142(15)*acd142(2)
      acd142(75)=acd142(26)*acd142(10)
      acd142(73)=-acd142(75)+acd142(73)-acd142(74)
      acd142(74)=-acd142(25)*acd142(73)
      acd142(75)=acd142(17)*acd142(2)
      acd142(76)=acd142(28)*acd142(6)
      acd142(77)=acd142(29)*acd142(10)
      acd142(75)=acd142(77)+acd142(75)+acd142(76)
      acd142(76)=acd142(27)*acd142(75)
      acd142(77)=acd142(7)*acd142(13)
      acd142(78)=acd142(28)*acd142(16)
      acd142(79)=acd142(31)*acd142(18)
      acd142(77)=acd142(79)+acd142(77)+acd142(78)
      acd142(78)=acd142(30)*acd142(77)
      acd142(79)=acd142(1)*acd142(2)
      acd142(80)=acd142(12)*acd142(13)
      acd142(79)=acd142(79)+acd142(80)
      acd142(79)=acd142(3)*acd142(79)
      acd142(80)=acd142(1)*acd142(10)
      acd142(81)=acd142(35)*acd142(13)
      acd142(80)=acd142(80)+acd142(81)
      acd142(80)=acd142(11)*acd142(80)
      acd142(81)=acd142(12)*acd142(18)
      acd142(82)=acd142(32)*acd142(2)
      acd142(81)=acd142(81)+acd142(82)
      acd142(81)=acd142(19)*acd142(81)
      acd142(82)=acd142(32)*acd142(10)
      acd142(83)=acd142(35)*acd142(18)
      acd142(82)=acd142(82)+acd142(83)
      acd142(82)=acd142(33)*acd142(82)
      acd142(83)=acd142(18)*acd142(23)
      acd142(84)=acd142(9)*acd142(13)
      acd142(83)=acd142(83)-acd142(84)
      acd142(84)=-acd142(34)*acd142(83)
      acd142(85)=acd142(10)*acd142(23)
      acd142(86)=acd142(21)*acd142(2)
      acd142(85)=acd142(85)-acd142(86)
      acd142(86)=-acd142(36)*acd142(85)
      acd142(64)=acd142(86)+acd142(84)+acd142(78)+acd142(76)+acd142(74)+acd142(&
      &72)+acd142(82)+acd142(81)+acd142(80)+acd142(79)+acd142(70)+acd142(68)+ac&
      &d142(66)+acd142(64)
      acd142(63)=acd142(39)*acd142(63)
      acd142(65)=acd142(40)*acd142(65)
      acd142(66)=-acd142(45)*acd142(67)
      acd142(67)=-acd142(47)*acd142(69)
      acd142(68)=-acd142(41)*acd142(71)
      acd142(69)=-acd142(42)*acd142(73)
      acd142(70)=acd142(43)*acd142(75)
      acd142(71)=acd142(44)*acd142(77)
      acd142(72)=acd142(39)*acd142(2)
      acd142(73)=acd142(40)*acd142(13)
      acd142(72)=acd142(72)+acd142(73)
      acd142(72)=acd142(3)*acd142(72)
      acd142(73)=acd142(39)*acd142(10)
      acd142(74)=acd142(47)*acd142(13)
      acd142(73)=acd142(73)+acd142(74)
      acd142(73)=acd142(11)*acd142(73)
      acd142(74)=acd142(40)*acd142(18)
      acd142(75)=acd142(45)*acd142(2)
      acd142(74)=acd142(74)+acd142(75)
      acd142(74)=acd142(19)*acd142(74)
      acd142(75)=acd142(45)*acd142(10)
      acd142(76)=acd142(47)*acd142(18)
      acd142(75)=acd142(75)+acd142(76)
      acd142(75)=acd142(33)*acd142(75)
      acd142(76)=-acd142(46)*acd142(83)
      acd142(77)=-acd142(48)*acd142(85)
      acd142(78)=acd142(38)*acd142(37)
      acd142(79)=acd142(49)*acd142(13)
      acd142(80)=acd142(50)*acd142(2)
      acd142(81)=acd142(51)*acd142(4)
      acd142(82)=acd142(52)*acd142(14)
      acd142(83)=acd142(54)*acd142(53)
      acd142(84)=acd142(55)*acd142(16)
      acd142(85)=acd142(56)*acd142(6)
      acd142(86)=acd142(57)*acd142(18)
      acd142(87)=acd142(58)*acd142(8)
      acd142(88)=acd142(59)*acd142(10)
      acd142(89)=acd142(61)*acd142(60)
      acd142(90)=acd142(62)*acd142(20)
      acd142(63)=acd142(90)+acd142(89)+acd142(88)+acd142(87)+acd142(86)+acd142(&
      &85)+acd142(84)+acd142(83)+acd142(82)+acd142(81)+acd142(80)+acd142(79)+2.&
      &0_ki*acd142(78)+acd142(77)+acd142(76)+acd142(71)+acd142(70)+acd142(69)+a&
      &cd142(68)+acd142(75)+acd142(74)+acd142(73)+acd142(72)+acd142(67)+acd142(&
      &66)+acd142(65)+acd142(63)
      brack(ninjaidxt0x0mu0)=acd142(63)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd142(64)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d142h4_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd142h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k5
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
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
end module     p2_gg_httbar_d142h4l132_qp
