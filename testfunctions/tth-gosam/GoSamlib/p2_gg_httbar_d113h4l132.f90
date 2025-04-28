module     p2_gg_httbar_d113h4l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d113h4l132.f90
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
      use p2_gg_httbar_abbrevd113h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(15) :: acd113
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd113(1)=dotproduct(ninjaE3,spvak2e1)
      acd113(2)=dotproduct(ninjaE3,spvae2k2)
      acd113(3)=dotproduct(ninjaE3,spvae1e2)
      acd113(4)=abb113(42)
      acd113(5)=dotproduct(ninjaE3,spvae2l4)
      acd113(6)=dotproduct(ninjaE3,spval5e1)
      acd113(7)=abb113(50)
      acd113(8)=dotproduct(ninjaE3,spvae1k2)
      acd113(9)=dotproduct(ninjaE3,spvak2e2)
      acd113(10)=dotproduct(ninjaE3,spvae2e1)
      acd113(11)=dotproduct(ninjaE3,spvae1l4)
      acd113(12)=dotproduct(ninjaE3,spval5e2)
      acd113(13)=acd113(2)*acd113(1)*acd113(4)
      acd113(14)=-acd113(6)*acd113(5)*acd113(7)
      acd113(13)=acd113(13)+acd113(14)
      acd113(13)=acd113(3)*acd113(13)
      acd113(14)=acd113(9)*acd113(8)*acd113(4)
      acd113(15)=-acd113(12)*acd113(11)*acd113(7)
      acd113(14)=acd113(15)+acd113(14)
      acd113(14)=acd113(10)*acd113(14)
      acd113(13)=acd113(14)+acd113(13)
      brack(ninjaidxt2x0mu0)=0.0_ki
      brack(ninjaidxt1x0mu0)=acd113(13)
      brack(ninjaidxt1x1mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd113h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(88) :: acd113
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd113(1)=dotproduct(ninjaA1,spvak2e2)
      acd113(2)=dotproduct(ninjaE3,spvae2e1)
      acd113(3)=dotproduct(ninjaE3,spvae1k2)
      acd113(4)=abb113(42)
      acd113(5)=dotproduct(ninjaA1,spvae2e1)
      acd113(6)=dotproduct(ninjaE3,spvak2e2)
      acd113(7)=dotproduct(ninjaE3,spvae1l4)
      acd113(8)=dotproduct(ninjaE3,spval5e2)
      acd113(9)=abb113(50)
      acd113(10)=dotproduct(ninjaA1,spvae1l4)
      acd113(11)=dotproduct(ninjaA1,spvae2k2)
      acd113(12)=dotproduct(ninjaE3,spvae1e2)
      acd113(13)=dotproduct(ninjaE3,spvak2e1)
      acd113(14)=dotproduct(ninjaA1,spvae1e2)
      acd113(15)=dotproduct(ninjaE3,spvae2k2)
      acd113(16)=dotproduct(ninjaE3,spval5e1)
      acd113(17)=dotproduct(ninjaE3,spvae2l4)
      acd113(18)=dotproduct(ninjaA1,spval5e1)
      acd113(19)=dotproduct(ninjaA1,spvak2e1)
      acd113(20)=dotproduct(ninjaA1,spvae1k2)
      acd113(21)=dotproduct(ninjaA1,spval5e2)
      acd113(22)=dotproduct(ninjaA1,spvae2l4)
      acd113(23)=dotproduct(k2,ninjaE3)
      acd113(24)=abb113(44)
      acd113(25)=abb113(15)
      acd113(26)=abb113(35)
      acd113(27)=abb113(22)
      acd113(28)=dotproduct(ninjaA0,ninjaE3)
      acd113(29)=abb113(34)
      acd113(30)=abb113(54)
      acd113(31)=abb113(17)
      acd113(32)=abb113(48)
      acd113(33)=abb113(16)
      acd113(34)=abb113(26)
      acd113(35)=dotproduct(ninjaA0,spvak2e2)
      acd113(36)=dotproduct(ninjaA0,spvae2e1)
      acd113(37)=dotproduct(ninjaA0,spvae1l4)
      acd113(38)=dotproduct(ninjaA0,spvae2k2)
      acd113(39)=dotproduct(ninjaA0,spvae1e2)
      acd113(40)=dotproduct(ninjaA0,spval5e1)
      acd113(41)=dotproduct(ninjaA0,spvak2e1)
      acd113(42)=dotproduct(ninjaA0,spvae1k2)
      acd113(43)=dotproduct(ninjaA0,spval5e2)
      acd113(44)=dotproduct(ninjaA0,spvae2l4)
      acd113(45)=abb113(7)
      acd113(46)=abb113(20)
      acd113(47)=abb113(51)
      acd113(48)=dotproduct(ninjaE3,spvak2l4)
      acd113(49)=abb113(45)
      acd113(50)=abb113(31)
      acd113(51)=abb113(40)
      acd113(52)=dotproduct(ninjaE3,spval3e2)
      acd113(53)=abb113(57)
      acd113(54)=dotproduct(ninjaE3,spval5k2)
      acd113(55)=abb113(8)
      acd113(56)=abb113(19)
      acd113(57)=abb113(25)
      acd113(58)=dotproduct(ninjaE3,spvae2l3)
      acd113(59)=abb113(56)
      acd113(60)=abb113(10)
      acd113(61)=abb113(52)
      acd113(62)=abb113(43)
      acd113(63)=abb113(55)
      acd113(64)=abb113(53)
      acd113(65)=abb113(11)
      acd113(66)=abb113(58)
      acd113(67)=abb113(46)
      acd113(68)=abb113(18)
      acd113(69)=acd113(3)*acd113(4)
      acd113(70)=acd113(1)*acd113(69)
      acd113(71)=acd113(8)*acd113(9)
      acd113(72)=-acd113(10)*acd113(71)
      acd113(73)=acd113(6)*acd113(4)
      acd113(74)=acd113(20)*acd113(73)
      acd113(75)=acd113(7)*acd113(9)
      acd113(76)=-acd113(21)*acd113(75)
      acd113(70)=acd113(76)+acd113(74)+acd113(72)+acd113(70)
      acd113(70)=acd113(2)*acd113(70)
      acd113(72)=acd113(13)*acd113(4)
      acd113(74)=acd113(11)*acd113(72)
      acd113(76)=acd113(17)*acd113(9)
      acd113(77)=-acd113(18)*acd113(76)
      acd113(78)=acd113(15)*acd113(4)
      acd113(79)=acd113(19)*acd113(78)
      acd113(80)=acd113(16)*acd113(9)
      acd113(81)=-acd113(22)*acd113(80)
      acd113(74)=acd113(81)+acd113(79)+acd113(77)+acd113(74)
      acd113(74)=acd113(12)*acd113(74)
      acd113(77)=acd113(71)*acd113(7)
      acd113(79)=acd113(73)*acd113(3)
      acd113(77)=acd113(77)-acd113(79)
      acd113(79)=-acd113(5)*acd113(77)
      acd113(81)=acd113(76)*acd113(16)
      acd113(82)=acd113(78)*acd113(13)
      acd113(81)=acd113(81)-acd113(82)
      acd113(82)=-acd113(14)*acd113(81)
      acd113(70)=acd113(79)+acd113(82)+acd113(74)+acd113(70)
      acd113(74)=acd113(24)*acd113(23)
      acd113(79)=2.0_ki*acd113(28)
      acd113(82)=acd113(29)*acd113(79)
      acd113(69)=acd113(35)*acd113(69)
      acd113(71)=-acd113(37)*acd113(71)
      acd113(73)=acd113(42)*acd113(73)
      acd113(75)=-acd113(43)*acd113(75)
      acd113(83)=acd113(45)*acd113(6)
      acd113(84)=acd113(47)*acd113(7)
      acd113(85)=acd113(49)*acd113(48)
      acd113(86)=acd113(50)*acd113(3)
      acd113(87)=acd113(51)*acd113(8)
      acd113(88)=acd113(53)*acd113(52)
      acd113(69)=acd113(88)+acd113(87)+acd113(86)+acd113(85)+acd113(84)+acd113(&
      &83)+acd113(75)+acd113(73)+acd113(71)+acd113(69)+acd113(82)+acd113(74)
      acd113(69)=acd113(2)*acd113(69)
      acd113(71)=acd113(25)*acd113(23)
      acd113(73)=acd113(31)*acd113(79)
      acd113(72)=acd113(38)*acd113(72)
      acd113(74)=-acd113(40)*acd113(76)
      acd113(75)=acd113(41)*acd113(78)
      acd113(76)=-acd113(44)*acd113(80)
      acd113(78)=acd113(56)*acd113(54)
      acd113(80)=acd113(60)*acd113(15)
      acd113(82)=acd113(61)*acd113(16)
      acd113(83)=acd113(62)*acd113(13)
      acd113(84)=acd113(63)*acd113(58)
      acd113(85)=acd113(64)*acd113(17)
      acd113(71)=acd113(85)+acd113(84)+acd113(83)+acd113(82)+acd113(80)+acd113(&
      &78)+acd113(76)+acd113(75)+acd113(74)+acd113(72)+acd113(73)+acd113(71)
      acd113(71)=acd113(12)*acd113(71)
      acd113(72)=acd113(30)*acd113(7)
      acd113(73)=acd113(32)*acd113(16)
      acd113(74)=-acd113(33)*acd113(13)
      acd113(75)=-acd113(34)*acd113(3)
      acd113(72)=acd113(75)+acd113(74)+acd113(73)+acd113(72)
      acd113(72)=acd113(79)*acd113(72)
      acd113(73)=acd113(55)*acd113(54)
      acd113(74)=acd113(57)*acd113(15)
      acd113(75)=acd113(59)*acd113(58)
      acd113(73)=acd113(75)+acd113(74)+acd113(73)
      acd113(73)=acd113(7)*acd113(73)
      acd113(74)=acd113(46)*acd113(6)
      acd113(75)=acd113(65)*acd113(48)
      acd113(76)=acd113(66)*acd113(52)
      acd113(74)=acd113(76)+acd113(75)+acd113(74)
      acd113(74)=acd113(16)*acd113(74)
      acd113(75)=acd113(26)*acd113(13)
      acd113(76)=acd113(27)*acd113(3)
      acd113(75)=acd113(76)+acd113(75)
      acd113(75)=acd113(23)*acd113(75)
      acd113(76)=-acd113(36)*acd113(77)
      acd113(77)=-acd113(39)*acd113(81)
      acd113(78)=acd113(67)*acd113(52)*acd113(13)
      acd113(79)=acd113(68)*acd113(58)*acd113(3)
      acd113(69)=acd113(79)+acd113(78)+acd113(76)+acd113(77)+acd113(71)+acd113(&
      &69)+acd113(72)+acd113(74)+acd113(73)+acd113(75)
      brack(ninjaidxt0x0mu0)=acd113(69)
      brack(ninjaidxt0x0mu2)=0.0_ki
      brack(ninjaidxt0x1mu0)=acd113(70)
      brack(ninjaidxt0x2mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d113h4_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd113h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2+k3+k5
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d113h4l132
