module     p2_gg_httbar_d71h8l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d71h8l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd71h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd71
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      brack(ninjaidxt1x0mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd71h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(92) :: acd71
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd71(1)=dotproduct(k2,ninjaE3)
      acd71(2)=dotproduct(e2,ninjaE3)
      acd71(3)=abb71(17)
      acd71(4)=dotproduct(ninjaE3,spvak2e2)
      acd71(5)=abb71(35)
      acd71(6)=dotproduct(l5,ninjaE3)
      acd71(7)=abb71(103)
      acd71(8)=dotproduct(ninjaE3,spvae2l5)
      acd71(9)=abb71(31)
      acd71(10)=dotproduct(ninjaA0,ninjaE3)
      acd71(11)=abb71(27)
      acd71(12)=dotproduct(ninjaE3,spvak2l5)
      acd71(13)=abb71(11)
      acd71(14)=dotproduct(ninjaE3,spvak2l4)
      acd71(15)=abb71(20)
      acd71(16)=dotproduct(ninjaE3,spvak2l3)
      acd71(17)=abb71(24)
      acd71(18)=dotproduct(ninjaE3,spval3l5)
      acd71(19)=abb71(28)
      acd71(20)=dotproduct(ninjaE3,spvae1l3)
      acd71(21)=abb71(126)
      acd71(22)=dotproduct(ninjaE3,spvae1l5)
      acd71(23)=abb71(75)
      acd71(24)=dotproduct(ninjaE3,spvak2k1)
      acd71(25)=abb71(39)
      acd71(26)=dotproduct(ninjaE3,spvak1l5)
      acd71(27)=abb71(42)
      acd71(28)=dotproduct(ninjaE3,spvak1l3)
      acd71(29)=abb71(44)
      acd71(30)=dotproduct(ninjaE3,spvak2e1)
      acd71(31)=abb71(100)
      acd71(32)=dotproduct(ninjaE3,spval3k1)
      acd71(33)=abb71(66)
      acd71(34)=dotproduct(ninjaE3,spval3e1)
      acd71(35)=abb71(67)
      acd71(36)=dotproduct(ninjaE3,spval3l4)
      acd71(37)=abb71(112)
      acd71(38)=dotproduct(ninjaE3,spval4l5)
      acd71(39)=abb71(90)
      acd71(40)=dotproduct(ninjaE3,spval4l3)
      acd71(41)=abb71(115)
      acd71(42)=abb71(48)
      acd71(43)=abb71(81)
      acd71(44)=dotproduct(ninjaE3,spvae1e2)
      acd71(45)=abb71(38)
      acd71(46)=dotproduct(ninjaE3,spvae2l4)
      acd71(47)=abb71(83)
      acd71(48)=dotproduct(ninjaE3,spvae2k1)
      acd71(49)=abb71(84)
      acd71(50)=dotproduct(ninjaE3,spvae2e1)
      acd71(51)=abb71(85)
      acd71(52)=dotproduct(ninjaE3,spval4e2)
      acd71(53)=abb71(102)
      acd71(54)=dotproduct(ninjaE3,spvak1e2)
      acd71(55)=abb71(109)
      acd71(56)=dotproduct(ninjaE3,spvae2k2)
      acd71(57)=abb71(13)
      acd71(58)=dotproduct(ninjaE3,spval5e2)
      acd71(59)=abb71(23)
      acd71(60)=abb71(40)
      acd71(61)=abb71(65)
      acd71(62)=dotproduct(ninjaE3,spval3e2)
      acd71(63)=abb71(32)
      acd71(64)=dotproduct(ninjaE3,spval5l4)
      acd71(65)=dotproduct(ninjaE3,spval5k1)
      acd71(66)=dotproduct(ninjaE3,spval5e1)
      acd71(67)=dotproduct(ninjaE3,spvak1k2)
      acd71(68)=abb71(45)
      acd71(69)=dotproduct(ninjaE3,spvae1k2)
      acd71(70)=abb71(62)
      acd71(71)=dotproduct(ninjaE3,spval4k2)
      acd71(72)=abb71(94)
      acd71(73)=dotproduct(ninjaE3,spvae2l3)
      acd71(74)=acd71(18)*acd71(19)
      acd71(75)=acd71(40)*acd71(41)
      acd71(76)=acd71(38)*acd71(39)
      acd71(77)=acd71(36)*acd71(37)
      acd71(78)=acd71(34)*acd71(35)
      acd71(79)=acd71(32)*acd71(33)
      acd71(80)=acd71(30)*acd71(31)
      acd71(81)=acd71(28)*acd71(29)
      acd71(82)=acd71(26)*acd71(27)
      acd71(83)=acd71(24)*acd71(25)
      acd71(84)=acd71(22)*acd71(23)
      acd71(85)=acd71(20)*acd71(21)
      acd71(86)=acd71(16)*acd71(17)
      acd71(87)=acd71(14)*acd71(15)
      acd71(88)=acd71(6)*acd71(7)
      acd71(89)=acd71(1)*acd71(3)
      acd71(90)=acd71(12)*acd71(13)
      acd71(91)=2.0_ki*acd71(10)
      acd71(92)=acd71(11)*acd71(91)
      acd71(74)=acd71(92)+acd71(90)+acd71(89)+acd71(88)+acd71(87)+acd71(86)+acd&
      &71(85)+acd71(84)+acd71(83)+acd71(82)+acd71(81)+acd71(80)+acd71(79)+acd71&
      &(78)+acd71(77)+acd71(76)+acd71(74)+acd71(75)
      acd71(74)=acd71(2)*acd71(74)
      acd71(75)=acd71(55)*acd71(54)
      acd71(76)=-acd71(53)*acd71(52)
      acd71(77)=-acd71(51)*acd71(50)
      acd71(78)=acd71(49)*acd71(48)
      acd71(79)=-acd71(47)*acd71(46)
      acd71(80)=acd71(45)*acd71(44)
      acd71(81)=acd71(8)*acd71(42)
      acd71(82)=acd71(4)*acd71(43)
      acd71(75)=acd71(82)+acd71(81)+acd71(80)+acd71(79)+acd71(78)+acd71(77)+acd&
      &71(75)+acd71(76)
      acd71(75)=acd71(75)*acd71(91)
      acd71(76)=acd71(16)*acd71(63)
      acd71(77)=acd71(55)*acd71(28)
      acd71(78)=-acd71(53)*acd71(40)
      acd71(79)=acd71(45)*acd71(20)
      acd71(76)=acd71(79)+acd71(78)+acd71(76)+acd71(77)
      acd71(76)=acd71(62)*acd71(76)
      acd71(77)=acd71(55)*acd71(26)
      acd71(78)=-acd71(53)*acd71(38)
      acd71(79)=acd71(45)*acd71(22)
      acd71(80)=acd71(12)*acd71(43)
      acd71(77)=acd71(80)+acd71(79)+acd71(77)+acd71(78)
      acd71(77)=acd71(58)*acd71(77)
      acd71(78)=acd71(30)*acd71(61)
      acd71(79)=acd71(24)*acd71(60)
      acd71(80)=acd71(14)*acd71(59)
      acd71(81)=-acd71(12)*acd71(57)
      acd71(78)=acd71(81)+acd71(80)+acd71(78)+acd71(79)
      acd71(78)=acd71(56)*acd71(78)
      acd71(79)=-acd71(6)*acd71(9)
      acd71(80)=-acd71(51)*acd71(66)
      acd71(81)=acd71(49)*acd71(65)
      acd71(82)=-acd71(47)*acd71(64)
      acd71(79)=acd71(82)+acd71(81)+acd71(79)+acd71(80)
      acd71(79)=acd71(8)*acd71(79)
      acd71(80)=acd71(71)*acd71(72)
      acd71(81)=acd71(69)*acd71(70)
      acd71(82)=acd71(67)*acd71(68)
      acd71(83)=acd71(1)*acd71(5)
      acd71(80)=acd71(83)+acd71(82)+acd71(80)+acd71(81)
      acd71(80)=acd71(4)*acd71(80)
      acd71(81)=-acd71(51)*acd71(34)
      acd71(82)=acd71(49)*acd71(32)
      acd71(83)=-acd71(47)*acd71(36)
      acd71(81)=acd71(83)+acd71(81)+acd71(82)
      acd71(81)=acd71(73)*acd71(81)
      acd71(74)=acd71(74)+acd71(75)+acd71(80)+acd71(79)+acd71(78)+acd71(77)+acd&
      &71(76)+acd71(81)
      brack(ninjaidxt0x0mu0)=acd71(74)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d71h8_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd71h8_qp
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
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d71h8l132_qp
