module     p2_gg_httbar_d27h12l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d27h12l132.f90
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
      use p2_gg_httbar_abbrevd27h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(31) :: acd27
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd27(1)=dotproduct(ninjaE3,spvak2e2)
      acd27(2)=dotproduct(ninjaE3,spvae2e1)
      acd27(3)=abb27(15)
      acd27(4)=dotproduct(ninjaE3,spvae2k2)
      acd27(5)=abb27(19)
      acd27(6)=dotproduct(ninjaE3,spvae2k1)
      acd27(7)=abb27(21)
      acd27(8)=dotproduct(ninjaE3,spvae2l4)
      acd27(9)=abb27(24)
      acd27(10)=dotproduct(ninjaE3,spval3e2)
      acd27(11)=abb27(34)
      acd27(12)=abb27(28)
      acd27(13)=abb27(20)
      acd27(14)=abb27(48)
      acd27(15)=dotproduct(ninjaE3,spval4e2)
      acd27(16)=dotproduct(ninjaE3,spvae2l5)
      acd27(17)=abb27(31)
      acd27(18)=dotproduct(ninjaE3,spvae2l3)
      acd27(19)=abb27(36)
      acd27(20)=dotproduct(ninjaE3,spvak1e2)
      acd27(21)=abb27(63)
      acd27(22)=dotproduct(ninjaE3,spvae1e2)
      acd27(23)=abb27(46)
      acd27(24)=abb27(45)
      acd27(25)=abb27(50)
      acd27(26)=acd27(3)*acd27(2)
      acd27(27)=acd27(5)*acd27(4)
      acd27(28)=acd27(7)*acd27(6)
      acd27(29)=acd27(9)*acd27(8)
      acd27(26)=acd27(29)+acd27(28)+acd27(26)+acd27(27)
      acd27(26)=acd27(1)*acd27(26)
      acd27(27)=-acd27(11)*acd27(2)
      acd27(28)=acd27(12)*acd27(4)
      acd27(29)=acd27(13)*acd27(6)
      acd27(30)=acd27(14)*acd27(8)
      acd27(27)=acd27(30)+acd27(29)+acd27(28)+acd27(27)
      acd27(27)=acd27(10)*acd27(27)
      acd27(28)=acd27(17)*acd27(15)
      acd27(29)=-acd27(21)*acd27(20)
      acd27(30)=acd27(23)*acd27(22)
      acd27(28)=acd27(30)+acd27(29)+acd27(28)
      acd27(28)=acd27(16)*acd27(28)
      acd27(29)=acd27(19)*acd27(15)
      acd27(30)=-acd27(24)*acd27(20)
      acd27(31)=acd27(25)*acd27(22)
      acd27(29)=acd27(31)+acd27(30)+acd27(29)
      acd27(29)=acd27(18)*acd27(29)
      acd27(26)=acd27(27)+acd27(26)+acd27(29)+acd27(28)
      brack(ninjaidxt1x0mu0)=acd27(26)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd27h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(81) :: acd27
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd27(1)=dotproduct(ninjaA1,spvae2l5)
      acd27(2)=dotproduct(ninjaE3,spvak1e2)
      acd27(3)=abb27(63)
      acd27(4)=dotproduct(ninjaE3,spvae1e2)
      acd27(5)=abb27(46)
      acd27(6)=dotproduct(ninjaE3,spval4e2)
      acd27(7)=abb27(31)
      acd27(8)=dotproduct(ninjaA1,spvak2e2)
      acd27(9)=dotproduct(ninjaE3,spvae2k1)
      acd27(10)=abb27(21)
      acd27(11)=dotproduct(ninjaE3,spvae2e1)
      acd27(12)=abb27(15)
      acd27(13)=dotproduct(ninjaE3,spvae2l4)
      acd27(14)=abb27(24)
      acd27(15)=dotproduct(ninjaE3,spvae2k2)
      acd27(16)=abb27(19)
      acd27(17)=dotproduct(ninjaA1,spvae2k1)
      acd27(18)=dotproduct(ninjaE3,spvak2e2)
      acd27(19)=dotproduct(ninjaE3,spval3e2)
      acd27(20)=abb27(20)
      acd27(21)=dotproduct(ninjaA1,spvae2e1)
      acd27(22)=abb27(34)
      acd27(23)=dotproduct(ninjaA1,spvak1e2)
      acd27(24)=dotproduct(ninjaE3,spvae2l5)
      acd27(25)=dotproduct(ninjaE3,spvae2l3)
      acd27(26)=abb27(45)
      acd27(27)=dotproduct(ninjaA1,spvae2l4)
      acd27(28)=abb27(48)
      acd27(29)=dotproduct(ninjaA1,spvae2k2)
      acd27(30)=abb27(28)
      acd27(31)=dotproduct(ninjaA1,spval3e2)
      acd27(32)=dotproduct(ninjaA1,spvae1e2)
      acd27(33)=abb27(50)
      acd27(34)=dotproduct(ninjaA1,spval4e2)
      acd27(35)=abb27(36)
      acd27(36)=dotproduct(ninjaA1,spvae2l3)
      acd27(37)=dotproduct(ninjaA0,spvae2l5)
      acd27(38)=dotproduct(ninjaA0,spvak2e2)
      acd27(39)=dotproduct(ninjaA0,spvae2k1)
      acd27(40)=dotproduct(ninjaA0,spvae2e1)
      acd27(41)=dotproduct(ninjaA0,spvak1e2)
      acd27(42)=dotproduct(ninjaA0,spvae2l4)
      acd27(43)=dotproduct(ninjaA0,spvae2k2)
      acd27(44)=dotproduct(ninjaA0,spval3e2)
      acd27(45)=dotproduct(ninjaA0,spvae1e2)
      acd27(46)=dotproduct(ninjaA0,spval4e2)
      acd27(47)=dotproduct(ninjaA0,spvae2l3)
      acd27(48)=abb27(11)
      acd27(49)=abb27(12)
      acd27(50)=abb27(13)
      acd27(51)=abb27(14)
      acd27(52)=abb27(16)
      acd27(53)=abb27(18)
      acd27(54)=abb27(41)
      acd27(55)=abb27(43)
      acd27(56)=abb27(25)
      acd27(57)=abb27(29)
      acd27(58)=abb27(30)
      acd27(59)=acd27(10)*acd27(9)
      acd27(60)=acd27(12)*acd27(11)
      acd27(61)=acd27(14)*acd27(13)
      acd27(62)=acd27(16)*acd27(15)
      acd27(59)=acd27(59)+acd27(60)+acd27(61)+acd27(62)
      acd27(60)=acd27(8)*acd27(59)
      acd27(61)=acd27(20)*acd27(9)
      acd27(62)=acd27(22)*acd27(11)
      acd27(63)=acd27(28)*acd27(13)
      acd27(64)=acd27(30)*acd27(15)
      acd27(61)=acd27(61)-acd27(62)+acd27(63)+acd27(64)
      acd27(62)=acd27(31)*acd27(61)
      acd27(63)=acd27(3)*acd27(2)
      acd27(64)=acd27(5)*acd27(4)
      acd27(65)=acd27(7)*acd27(6)
      acd27(63)=-acd27(65)+acd27(63)-acd27(64)
      acd27(64)=-acd27(1)*acd27(63)
      acd27(65)=acd27(26)*acd27(2)
      acd27(66)=acd27(33)*acd27(4)
      acd27(67)=acd27(35)*acd27(6)
      acd27(65)=-acd27(67)+acd27(65)-acd27(66)
      acd27(66)=-acd27(36)*acd27(65)
      acd27(67)=acd27(10)*acd27(18)
      acd27(68)=acd27(20)*acd27(19)
      acd27(67)=acd27(67)+acd27(68)
      acd27(68)=acd27(17)*acd27(67)
      acd27(69)=acd27(12)*acd27(18)
      acd27(70)=acd27(22)*acd27(19)
      acd27(69)=acd27(69)-acd27(70)
      acd27(70)=acd27(21)*acd27(69)
      acd27(71)=acd27(3)*acd27(24)
      acd27(72)=acd27(26)*acd27(25)
      acd27(71)=acd27(71)+acd27(72)
      acd27(72)=-acd27(23)*acd27(71)
      acd27(73)=acd27(14)*acd27(18)
      acd27(74)=acd27(28)*acd27(19)
      acd27(73)=acd27(73)+acd27(74)
      acd27(74)=acd27(27)*acd27(73)
      acd27(75)=acd27(16)*acd27(18)
      acd27(76)=acd27(30)*acd27(19)
      acd27(75)=acd27(75)+acd27(76)
      acd27(76)=acd27(29)*acd27(75)
      acd27(77)=acd27(5)*acd27(24)
      acd27(78)=acd27(33)*acd27(25)
      acd27(77)=acd27(77)+acd27(78)
      acd27(78)=acd27(32)*acd27(77)
      acd27(79)=acd27(7)*acd27(24)
      acd27(80)=acd27(35)*acd27(25)
      acd27(79)=acd27(79)+acd27(80)
      acd27(80)=acd27(34)*acd27(79)
      acd27(60)=acd27(80)+acd27(78)+acd27(76)+acd27(74)+acd27(72)+acd27(70)+acd&
      &27(68)+acd27(66)+acd27(64)+acd27(62)+acd27(60)
      acd27(59)=acd27(38)*acd27(59)
      acd27(61)=acd27(44)*acd27(61)
      acd27(62)=-acd27(37)*acd27(63)
      acd27(63)=-acd27(47)*acd27(65)
      acd27(64)=acd27(39)*acd27(67)
      acd27(65)=acd27(40)*acd27(69)
      acd27(66)=-acd27(41)*acd27(71)
      acd27(67)=acd27(42)*acd27(73)
      acd27(68)=acd27(43)*acd27(75)
      acd27(69)=acd27(45)*acd27(77)
      acd27(70)=acd27(46)*acd27(79)
      acd27(71)=acd27(48)*acd27(24)
      acd27(72)=acd27(49)*acd27(18)
      acd27(73)=acd27(50)*acd27(9)
      acd27(74)=acd27(51)*acd27(11)
      acd27(75)=acd27(52)*acd27(2)
      acd27(76)=acd27(53)*acd27(13)
      acd27(77)=acd27(54)*acd27(15)
      acd27(78)=acd27(55)*acd27(19)
      acd27(79)=acd27(56)*acd27(4)
      acd27(80)=acd27(57)*acd27(6)
      acd27(81)=acd27(58)*acd27(25)
      acd27(59)=acd27(81)+acd27(80)+acd27(79)+acd27(78)+acd27(77)+acd27(76)+acd&
      &27(75)+acd27(74)+acd27(73)+acd27(72)+acd27(71)+acd27(70)+acd27(69)+acd27&
      &(68)+acd27(67)+acd27(66)+acd27(65)+acd27(64)+acd27(63)+acd27(62)+acd27(5&
      &9)+acd27(61)
      brack(ninjaidxt0x0mu0)=acd27(59)
      brack(ninjaidxt0x1mu0)=acd27(60)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d27h12_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd27h12
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
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d27h12l132
