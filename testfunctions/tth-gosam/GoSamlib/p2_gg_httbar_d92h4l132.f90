module     p2_gg_httbar_d92h4l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d92h4l132.f90
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
      use p2_gg_httbar_abbrevd92h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(14) :: acd92
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd92(1)=dotproduct(e1,ninjaE3)
      acd92(2)=dotproduct(ninjaE3,spvak2e2)
      acd92(3)=dotproduct(ninjaE3,spvae2l3)
      acd92(4)=abb92(37)
      acd92(5)=dotproduct(ninjaE3,spvae2k2)
      acd92(6)=abb92(49)
      acd92(7)=dotproduct(ninjaE3,spval3e2)
      acd92(8)=dotproduct(ninjaE3,spvae2l4)
      acd92(9)=abb92(70)
      acd92(10)=dotproduct(ninjaE3,spval5e2)
      acd92(11)=abb92(72)
      acd92(12)=acd92(9)*acd92(7)
      acd92(13)=acd92(11)*acd92(10)
      acd92(12)=acd92(13)+acd92(12)
      acd92(12)=acd92(12)*acd92(8)
      acd92(13)=acd92(4)*acd92(3)
      acd92(14)=acd92(6)*acd92(5)
      acd92(13)=acd92(13)+acd92(14)
      acd92(13)=acd92(13)*acd92(2)
      acd92(12)=acd92(13)+acd92(12)
      acd92(12)=acd92(1)*acd92(12)
      brack(ninjaidxt1x0mu0)=acd92(12)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd92h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(82) :: acd92
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd92(1)=dotproduct(e1,ninjaA1)
      acd92(2)=dotproduct(ninjaE3,spvae2k2)
      acd92(3)=dotproduct(ninjaE3,spvak2e2)
      acd92(4)=abb92(49)
      acd92(5)=dotproduct(ninjaE3,spval5e2)
      acd92(6)=dotproduct(ninjaE3,spvae2l4)
      acd92(7)=abb92(72)
      acd92(8)=dotproduct(ninjaE3,spvae2l3)
      acd92(9)=abb92(37)
      acd92(10)=dotproduct(ninjaE3,spval3e2)
      acd92(11)=abb92(70)
      acd92(12)=dotproduct(e1,ninjaE3)
      acd92(13)=dotproduct(ninjaA1,spvae2k2)
      acd92(14)=dotproduct(ninjaA1,spval5e2)
      acd92(15)=dotproduct(ninjaA1,spvae2l3)
      acd92(16)=dotproduct(ninjaA1,spvak2e2)
      acd92(17)=dotproduct(ninjaA1,spvae2l4)
      acd92(18)=dotproduct(ninjaA1,spval3e2)
      acd92(19)=dotproduct(k2,ninjaE3)
      acd92(20)=dotproduct(ninjaE3,spvae1e2)
      acd92(21)=abb92(90)
      acd92(22)=dotproduct(ninjaE3,spvae2e1)
      acd92(23)=abb92(46)
      acd92(24)=dotproduct(l4,ninjaE3)
      acd92(25)=abb92(56)
      acd92(26)=dotproduct(e1,ninjaA0)
      acd92(27)=dotproduct(ninjaA0,spvae2k2)
      acd92(28)=dotproduct(ninjaA0,spval5e2)
      acd92(29)=dotproduct(ninjaA0,spvae2l3)
      acd92(30)=dotproduct(ninjaA0,spvak2e2)
      acd92(31)=dotproduct(ninjaA0,spvae2l4)
      acd92(32)=dotproduct(ninjaA0,spval3e2)
      acd92(33)=abb92(10)
      acd92(34)=abb92(11)
      acd92(35)=abb92(14)
      acd92(36)=abb92(22)
      acd92(37)=abb92(48)
      acd92(38)=abb92(54)
      acd92(39)=dotproduct(ninjaA0,ninjaE3)
      acd92(40)=abb92(78)
      acd92(41)=abb92(65)
      acd92(42)=abb92(83)
      acd92(43)=abb92(80)
      acd92(44)=abb92(77)
      acd92(45)=dotproduct(ninjaE3,spval5k2)
      acd92(46)=abb92(17)
      acd92(47)=dotproduct(ninjaE3,spvak2l4)
      acd92(48)=abb92(26)
      acd92(49)=dotproduct(ninjaE3,spvae2k1)
      acd92(50)=abb92(24)
      acd92(51)=dotproduct(ninjaE3,spval5l4)
      acd92(52)=abb92(28)
      acd92(53)=dotproduct(ninjaE3,spval3l4)
      acd92(54)=abb92(44)
      acd92(55)=abb92(59)
      acd92(56)=dotproduct(ninjaE3,spval3k2)
      acd92(57)=abb92(55)
      acd92(58)=abb92(18)
      acd92(59)=dotproduct(ninjaE3,spvak1e2)
      acd92(60)=abb92(21)
      acd92(61)=abb92(31)
      acd92(62)=dotproduct(ninjaE3,spvak2l3)
      acd92(63)=abb92(35)
      acd92(64)=dotproduct(ninjaE3,spval4l3)
      acd92(65)=abb92(36)
      acd92(66)=dotproduct(ninjaE3,spval4k2)
      acd92(67)=abb92(43)
      acd92(68)=abb92(52)
      acd92(69)=acd92(10)*acd92(11)
      acd92(70)=acd92(5)*acd92(7)
      acd92(69)=acd92(69)+acd92(70)
      acd92(70)=acd92(17)*acd92(69)
      acd92(71)=acd92(8)*acd92(9)
      acd92(72)=acd92(2)*acd92(4)
      acd92(71)=acd92(71)+acd92(72)
      acd92(72)=acd92(16)*acd92(71)
      acd92(73)=acd92(11)*acd92(18)
      acd92(74)=acd92(7)*acd92(14)
      acd92(73)=acd92(73)+acd92(74)
      acd92(73)=acd92(6)*acd92(73)
      acd92(74)=acd92(9)*acd92(15)
      acd92(75)=acd92(4)*acd92(13)
      acd92(74)=acd92(74)+acd92(75)
      acd92(74)=acd92(3)*acd92(74)
      acd92(70)=acd92(74)+acd92(73)+acd92(72)+acd92(70)
      acd92(70)=acd92(12)*acd92(70)
      acd92(69)=acd92(69)*acd92(6)
      acd92(71)=acd92(71)*acd92(3)
      acd92(69)=acd92(69)+acd92(71)
      acd92(71)=acd92(1)*acd92(69)
      acd92(70)=acd92(70)+acd92(71)
      acd92(71)=acd92(66)*acd92(67)
      acd92(72)=acd92(64)*acd92(65)
      acd92(73)=acd92(62)*acd92(63)
      acd92(74)=acd92(59)*acd92(60)
      acd92(75)=-acd92(24)*acd92(25)
      acd92(76)=acd92(47)*acd92(58)
      acd92(77)=2.0_ki*acd92(39)
      acd92(78)=acd92(41)*acd92(77)
      acd92(79)=acd92(19)*acd92(23)
      acd92(80)=acd92(10)*acd92(68)
      acd92(81)=acd92(5)*acd92(43)
      acd92(82)=acd92(3)*acd92(61)
      acd92(71)=acd92(82)+acd92(81)+acd92(80)+acd92(79)+acd92(78)+acd92(76)+acd&
      &92(75)+acd92(74)+acd92(73)+acd92(71)+acd92(72)
      acd92(71)=acd92(22)*acd92(71)
      acd92(72)=acd92(56)*acd92(57)
      acd92(73)=acd92(53)*acd92(54)
      acd92(74)=acd92(51)*acd92(52)
      acd92(75)=acd92(49)*acd92(50)
      acd92(76)=acd92(45)*acd92(46)
      acd92(78)=acd92(47)*acd92(48)
      acd92(77)=acd92(40)*acd92(77)
      acd92(79)=acd92(19)*acd92(21)
      acd92(80)=acd92(8)*acd92(44)
      acd92(81)=acd92(2)*acd92(42)
      acd92(82)=acd92(6)*acd92(55)
      acd92(72)=acd92(82)+acd92(81)+acd92(80)+acd92(79)+acd92(77)+acd92(78)+acd&
      &92(76)+acd92(75)+acd92(74)+acd92(72)+acd92(73)
      acd92(72)=acd92(20)*acd92(72)
      acd92(73)=acd92(11)*acd92(32)
      acd92(74)=acd92(7)*acd92(28)
      acd92(73)=acd92(74)+acd92(37)+acd92(73)
      acd92(73)=acd92(6)*acd92(73)
      acd92(74)=acd92(9)*acd92(29)
      acd92(75)=acd92(4)*acd92(27)
      acd92(74)=acd92(75)+acd92(36)+acd92(74)
      acd92(74)=acd92(3)*acd92(74)
      acd92(75)=acd92(11)*acd92(31)
      acd92(75)=acd92(38)+acd92(75)
      acd92(75)=acd92(10)*acd92(75)
      acd92(76)=acd92(9)*acd92(30)
      acd92(76)=acd92(35)+acd92(76)
      acd92(76)=acd92(8)*acd92(76)
      acd92(77)=acd92(7)*acd92(31)
      acd92(77)=acd92(34)+acd92(77)
      acd92(77)=acd92(5)*acd92(77)
      acd92(78)=acd92(4)*acd92(30)
      acd92(78)=acd92(33)+acd92(78)
      acd92(78)=acd92(2)*acd92(78)
      acd92(73)=acd92(74)+acd92(73)+acd92(78)+acd92(77)+acd92(75)+acd92(76)
      acd92(73)=acd92(12)*acd92(73)
      acd92(69)=acd92(26)*acd92(69)
      acd92(69)=acd92(73)+acd92(72)+acd92(71)+acd92(69)
      brack(ninjaidxt0x0mu0)=acd92(69)
      brack(ninjaidxt0x1mu0)=acd92(70)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d92h4_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd92h4
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
end module     p2_gg_httbar_d92h4l132
