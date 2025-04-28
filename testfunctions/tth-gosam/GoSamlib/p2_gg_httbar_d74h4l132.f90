module     p2_gg_httbar_d74h4l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d74h4l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd74h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd74
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd74h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(92) :: acd74
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd74(1)=dotproduct(k2,ninjaE3)
      acd74(2)=dotproduct(e2,ninjaE3)
      acd74(3)=abb74(17)
      acd74(4)=dotproduct(ninjaE3,spvak2e2)
      acd74(5)=abb74(35)
      acd74(6)=dotproduct(l4,ninjaE3)
      acd74(7)=abb74(103)
      acd74(8)=dotproduct(ninjaE3,spvae2l4)
      acd74(9)=abb74(31)
      acd74(10)=dotproduct(ninjaA0,ninjaE3)
      acd74(11)=abb74(27)
      acd74(12)=dotproduct(ninjaE3,spvak2l4)
      acd74(13)=abb74(11)
      acd74(14)=dotproduct(ninjaE3,spvak2l3)
      acd74(15)=abb74(20)
      acd74(16)=dotproduct(ninjaE3,spvak2k1)
      acd74(17)=abb74(24)
      acd74(18)=dotproduct(ninjaE3,spval3l4)
      acd74(19)=abb74(28)
      acd74(20)=dotproduct(ninjaE3,spvae1l3)
      acd74(21)=abb74(126)
      acd74(22)=dotproduct(ninjaE3,spvae1l4)
      acd74(23)=abb74(97)
      acd74(24)=dotproduct(ninjaE3,spvak1l4)
      acd74(25)=abb74(39)
      acd74(26)=dotproduct(ninjaE3,spvak1l3)
      acd74(27)=abb74(42)
      acd74(28)=dotproduct(ninjaE3,spval3k1)
      acd74(29)=abb74(47)
      acd74(30)=dotproduct(ninjaE3,spvak2l5)
      acd74(31)=abb74(66)
      acd74(32)=dotproduct(ninjaE3,spvak2e1)
      acd74(33)=abb74(100)
      acd74(34)=dotproduct(ninjaE3,spval3e1)
      acd74(35)=abb74(67)
      acd74(36)=dotproduct(ninjaE3,spval5l3)
      acd74(37)=abb74(123)
      acd74(38)=dotproduct(ninjaE3,spval5l4)
      acd74(39)=abb74(115)
      acd74(40)=dotproduct(ninjaE3,spval3l5)
      acd74(41)=abb74(112)
      acd74(42)=abb74(53)
      acd74(43)=abb74(81)
      acd74(44)=dotproduct(ninjaE3,spvae1e2)
      acd74(45)=abb74(38)
      acd74(46)=dotproduct(ninjaE3,spval5e2)
      acd74(47)=abb74(75)
      acd74(48)=dotproduct(ninjaE3,spvae2l5)
      acd74(49)=abb74(83)
      acd74(50)=dotproduct(ninjaE3,spvae2k1)
      acd74(51)=abb74(84)
      acd74(52)=dotproduct(ninjaE3,spvae2e1)
      acd74(53)=abb74(85)
      acd74(54)=dotproduct(ninjaE3,spvak1e2)
      acd74(55)=abb74(109)
      acd74(56)=dotproduct(ninjaE3,spvae2k2)
      acd74(57)=abb74(13)
      acd74(58)=dotproduct(ninjaE3,spval4e2)
      acd74(59)=abb74(32)
      acd74(60)=abb74(54)
      acd74(61)=abb74(65)
      acd74(62)=dotproduct(ninjaE3,spval3e2)
      acd74(63)=abb74(21)
      acd74(64)=dotproduct(ninjaE3,spval4l5)
      acd74(65)=dotproduct(ninjaE3,spval4k1)
      acd74(66)=dotproduct(ninjaE3,spval4e1)
      acd74(67)=dotproduct(ninjaE3,spvak1k2)
      acd74(68)=abb74(44)
      acd74(69)=dotproduct(ninjaE3,spvae1k2)
      acd74(70)=abb74(62)
      acd74(71)=dotproduct(ninjaE3,spval5k2)
      acd74(72)=abb74(94)
      acd74(73)=dotproduct(ninjaE3,spvae2l3)
      acd74(74)=acd74(18)*acd74(19)
      acd74(75)=acd74(40)*acd74(41)
      acd74(76)=acd74(38)*acd74(39)
      acd74(77)=acd74(36)*acd74(37)
      acd74(78)=acd74(34)*acd74(35)
      acd74(79)=acd74(32)*acd74(33)
      acd74(80)=acd74(30)*acd74(31)
      acd74(81)=acd74(28)*acd74(29)
      acd74(82)=acd74(26)*acd74(27)
      acd74(83)=acd74(24)*acd74(25)
      acd74(84)=acd74(22)*acd74(23)
      acd74(85)=acd74(20)*acd74(21)
      acd74(86)=acd74(16)*acd74(17)
      acd74(87)=acd74(14)*acd74(15)
      acd74(88)=acd74(6)*acd74(7)
      acd74(89)=acd74(1)*acd74(3)
      acd74(90)=acd74(12)*acd74(13)
      acd74(91)=2.0_ki*acd74(10)
      acd74(92)=acd74(11)*acd74(91)
      acd74(74)=acd74(92)+acd74(90)+acd74(89)+acd74(88)+acd74(87)+acd74(86)+acd&
      &74(85)+acd74(84)+acd74(83)+acd74(82)+acd74(81)+acd74(80)+acd74(79)+acd74&
      &(78)+acd74(77)+acd74(76)+acd74(74)+acd74(75)
      acd74(74)=acd74(2)*acd74(74)
      acd74(75)=-acd74(55)*acd74(54)
      acd74(76)=acd74(53)*acd74(52)
      acd74(77)=-acd74(51)*acd74(50)
      acd74(78)=acd74(49)*acd74(48)
      acd74(79)=acd74(47)*acd74(46)
      acd74(80)=-acd74(45)*acd74(44)
      acd74(81)=acd74(8)*acd74(42)
      acd74(82)=-acd74(4)*acd74(43)
      acd74(75)=acd74(82)+acd74(81)+acd74(80)+acd74(79)+acd74(78)+acd74(77)+acd&
      &74(75)+acd74(76)
      acd74(75)=acd74(75)*acd74(91)
      acd74(76)=acd74(14)*acd74(63)
      acd74(77)=-acd74(55)*acd74(26)
      acd74(78)=acd74(47)*acd74(36)
      acd74(79)=-acd74(45)*acd74(20)
      acd74(76)=acd74(79)+acd74(78)+acd74(76)+acd74(77)
      acd74(76)=acd74(62)*acd74(76)
      acd74(77)=-acd74(55)*acd74(24)
      acd74(78)=acd74(47)*acd74(38)
      acd74(79)=-acd74(45)*acd74(22)
      acd74(80)=-acd74(12)*acd74(43)
      acd74(77)=acd74(80)+acd74(79)+acd74(77)+acd74(78)
      acd74(77)=acd74(58)*acd74(77)
      acd74(78)=acd74(32)*acd74(61)
      acd74(79)=acd74(30)*acd74(60)
      acd74(80)=acd74(16)*acd74(59)
      acd74(81)=acd74(12)*acd74(57)
      acd74(78)=acd74(81)+acd74(80)+acd74(78)+acd74(79)
      acd74(78)=acd74(56)*acd74(78)
      acd74(79)=acd74(6)*acd74(9)
      acd74(80)=acd74(53)*acd74(66)
      acd74(81)=-acd74(51)*acd74(65)
      acd74(82)=acd74(49)*acd74(64)
      acd74(79)=acd74(82)+acd74(81)+acd74(79)+acd74(80)
      acd74(79)=acd74(8)*acd74(79)
      acd74(80)=acd74(71)*acd74(72)
      acd74(81)=acd74(69)*acd74(70)
      acd74(82)=acd74(67)*acd74(68)
      acd74(83)=acd74(1)*acd74(5)
      acd74(80)=acd74(83)+acd74(82)+acd74(80)+acd74(81)
      acd74(80)=acd74(4)*acd74(80)
      acd74(81)=acd74(53)*acd74(34)
      acd74(82)=-acd74(51)*acd74(28)
      acd74(83)=acd74(49)*acd74(40)
      acd74(81)=acd74(83)+acd74(81)+acd74(82)
      acd74(81)=acd74(73)*acd74(81)
      acd74(74)=acd74(74)+acd74(75)+acd74(80)+acd74(79)+acd74(78)+acd74(77)+acd&
      &74(76)+acd74(81)
      brack(ninjaidxt0x0mu0)=acd74(74)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d74h4_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd74h4
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
end module     p2_gg_httbar_d74h4l132
